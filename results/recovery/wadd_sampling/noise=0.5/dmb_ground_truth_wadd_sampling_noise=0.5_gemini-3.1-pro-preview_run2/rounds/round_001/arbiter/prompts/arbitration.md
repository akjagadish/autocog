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
Weighted Additive (WADD) Theory with Non-Linear Cue Scaling: People evaluate multi-attribute options by computing an overall value for each option. This value is determined by taking a weighted sum of the option's features, where the weights correspond to the subjective validities of the respective cues scaled by a non-linear parameter. This scaling allows decision-makers to flexibly upweight highly valid cues (approaching a non-compensatory strategy) or downweight them (approaching an equal-weight tallying strategy). Decision-makers then choose probabilistically between the options by comparing these total weighted scores via a softmax function.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities
    weights = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for both options
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## THEORY 2 — pi_2
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 8: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]

**Rationale:** To cleanly dissociate WADD (Weighted Additive with non-linear scaling) from Tallying, we use 5 features with a wide spread of validities. Tallying purely counts the number of winning features, completely ignoring the validities (treating all features equally). WADD computes a weighted sum, where weights are validities scaled by a non-linear parameter gamma. We design trials that fall into three categories: (1) Ties in Tallying where WADD has a strong preference because it respects the validities of the winning features; (2) Contradictions where Tallying prefers an option with a higher quantity of low-validity features, while WADD prefers the option with fewer but much higher-validity features (especially for gamma > 1); (3) Alignments with different confidence levels. This thoroughly isolates the feature-counting mechanism of Tallying from the validity-weighted compensatory mechanism of WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where Tallying predicts a perfect tie because both options have the exact same number of positive features (Trials 1, 2, and 3). In these specific trials, Option A is explicitly designed to have features with higher validities than Option B. Tallying completely ignores these validities and will guess at random, resulting in a choice proportion of 0.5 for Option A. WADD, however, respects the validities and will consistently prefer Option A, resulting in a choice proportion significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Calculate the total number of positive features for each option
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where Tallying sees a tie (equal number of features)
    tied_trials = sum_a == sum_b
    
    if tied_trials.sum() == 0:
        return 0.5
        
    # Calculate the proportion of times Option A was chosen in these tied trials
    # (Response 0 means Option A was chosen)
    prob_a_chosen = (data.loc[tied_trials, 'response'] == 0).mean()
    
    return float(prob_a_chosen)

### RESULTS
- Predicted under pi_3 (simulated): 0.8378 (var=0.0112)
- Predicted under pi_2 (simulated): 0.4872 (var=0.0059)
- Observed on real data: 0.5611 (var=0.0131)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** To effectively dissociate Tallying from WADD, we must exploit Tallying's strict unweighted counting mechanism against WADD's validity-weighted (and non-linearly scalable) compensatory mechanism. We use 5 features with a wide spread of validities. The trials are crafted to produce three types of dissociations: (1) 'Tallying Ties' where both options have an equal number of positive features (Tallying predicts guessing), but WADD strongly prefers the option with higher-validity features; (2) 'Quantity vs. Quality' contradictions where Tallying prefers the option with more positive features (e.g., 3 low-validity vs. 2 high-validity), while WADD prefers the option with fewer but more predictive features; (3) 'Non-linear WADD shifts' where Tallying firmly predicts one option, but WADD's preference flips depending on the gamma parameter (e.g., heavily upweighting the single best cue).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
We exploit a strict symmetry in Tallying that is broken by WADD's non-linear cue scaling. In Trial 4, Tallying sees 2 wins for A and 3 for B (difference of -1). In Trial 5, Tallying sees 3 wins for A and 2 for B (difference of +1). Because the win differences are exactly symmetric (-1 and +1), Tallying predicts that the probability of choosing A in Trial 4 and Trial 5 will sum exactly to 1.0, completely independent of the softmax temperature (beta) or lapse rate (epsilon). 

For WADD, however, the sum of the expected value differences for these two trials is 2*(V1 - V3). Since the first feature's validity (0.95) is much larger than the third's (0.75), and this gap is further amplified by the non-linear scaling parameter gamma, WADD overwhelmingly favors A in Trial 5, and often also favors A in Trial 4 (when gamma is large enough to make the top two features outweigh the bottom three). As a result, WADD predicts the sum of these probabilities to be substantially greater than 1.0. This metric will yield ~1.0 with very low variance for Tallying, and a significantly higher value for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    # Create a string representation of option A's features to identify trials safely
    data['A_str'] = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # Trial 4: A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    p_a_t4 = 1.0 - data.loc[data['A_str'] == '11000', 'response'].mean()
    
    # Trial 5: A=[1, 0, 0, 1, 1] vs B=[0, 1, 1, 0, 0]
    p_a_t5 = 1.0 - data.loc[data['A_str'] == '10011', 'response'].mean()
    
    # Return the sum of probabilities of choosing A in these two trials
    return float(p_a_t4 + p_a_t5)

### RESULTS
- Predicted under pi_3 (simulated): 1.4888 (var=0.0953)
- Predicted under pi_2 (simulated): 0.9800 (var=0.0141)
- Observed on real data: 1.0350 (var=0.0694)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    # The validities are monotonically decreasing from left to right.
    # Thus, Take The Best (TTB) will choose based on the first feature
    # (from index 0 to 4) that discriminates between A and B.
    diff = a_ratings - b_ratings
    
    # Find the index of the first discriminating cue (where difference is non-zero)
    first_diff_idx = (diff != 0).argmax(axis=1)
    
    # Extract the difference value at that first discriminating cue
    first_diff_val = np.take_along_axis(diff, first_diff_idx[:, None], axis=1).squeeze()
    
    # If first_diff_val > 0 (A has the feature, B does not), TTB predicts A (response 0).
    # If first_diff_val < 0 (B has the feature, A does not), TTB predicts B (response 1).
    ttb_predictions = (first_diff_val < 0).astype(int)
    
    # Return the proportion of choices that match the TTB prediction.
    return float(np.mean(responses == ttb_predictions))
```

**Observed (real) value:** 0.4167 (var=0.0087)
**Predicted under pi_3:** 0.4692 (var=0.0436)
**Predicted under pi_2:** 0.1467 (var=0.0087)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    diff = a_sums - b_sums
    mask = diff != 0
    
    if not mask.any():
        return 0.5
        
    tallying_pred = (diff < 0).astype(int)
    accuracy = (data.loc[mask, 'response'] == tallying_pred[mask]).mean()
    
    return float(accuracy)
```

**Observed (real) value:** 0.6044 (var=0.0126)
**Predicted under pi_3:** 0.7206 (var=0.0372)
**Predicted under pi_2:** 0.8578 (var=0.0079)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, neither pure Tallying (Theory 2) nor non-linearly scaled WADD (Theory 1) consistently captures the observed data. In Experiments 1 and 2, the observed metrics (0.56 and 1.03) are much closer to Tallying's predictions (0.49 and 0.98) than WADD's (0.84 and 1.49), indicating that subjects largely ignore validities when tallying feature wins. However, in Experiments 3 and 4, Tallying fails dramatically: it drastically underpredicts TTB-consistent choices in Exp 3 (0.15 predicted vs. 0.42 observed) and overpredicts Tallying-consistent choices in Exp 4 (0.86 predicted vs. 0.60 observed). WADD performs better on these latter experiments but fundamentally fails the dissociation tests in Exps 1 and 2. This suggests that human decision-makers are likely using a heterogeneous or hybrid approach\u2014perhaps a strategy selection mechanism mixing Tallying with a validity-sensitive strategy (like Take-The-Best), or relying on rank-based weights rather than absolute validities.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a new 'Strategy Mixture' or 'Hybrid Heuristic' theory. Instead of assuming all subjects use pure Tallying or pure WADD, this theory should propose that decision-makers probabilistically choose between a compensatory/validity-based strategy (like Take-The-Best or linear WADD) and a non-compensatory counting strategy (Tallying). Alternatively, propose a model where cue weights are based on their ordinal rank rather than their absolute validities, allowing for a moderate sensitivity to cue importance that doesn't completely overwhelm feature counts."
}
```

## Usage

```json
{
  "prompt_token_count": 5415,
  "candidates_token_count": 411,
  "total_token_count": 6807
}
```
