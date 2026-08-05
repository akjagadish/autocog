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
People evaluate options by taking a weighted sum of their feature values, where the weights correspond to the subjective validities of the cues (Weighted Additive rule). This allows for fully compensatory decision making, where many small advantages can outweigh a single large advantage, and is sensitive to the cardinal magnitude of feature differences, unlike simple counting (Tallying) or one-reason heuristics (Take The Best).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    if len(val) != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
    
    # Compute the weighted sum of feature values for each option.
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    return int(np.argmax(probabilities))


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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[5, 5, 6, 6]  B=[10, 10, 0, 0]
  trial 2: A=[0, 0, 8, 8]  B=[2, 2, 0, 0]
  trial 3: A=[10, 0, 0, 0]  B=[0, 2, 2, 2]
  trial 4: A=[1, 1, 1, 1]  B=[0, 0, 0, 10]
  trial 5: A=[10, 0, 0, 0]  B=[7, 1, 1, 1]
  trial 6: A=[0, 10, 0, 0]  B=[1, 0, 1, 1]

**Rationale:** This design strictly dissociates the Weighted Additive (WADD) rule from Tallying by manipulating both cardinal feature magnitudes and feature validities. Tallying relies solely on counting the number of features where one option strictly beats the other, ignoring the size of the difference and the validities of the features. WADD, in contrast, computes a validity-weighted sum of the cardinal values. The trials are constructed such that in some cases Tallying predicts a tie (because both options win on an equal number of features) while WADD predicts a strong preference due to magnitude differences. In other cases, Tallying strongly prefers one option because it wins on more features (e.g., 3 wins vs 1 win), but WADD prefers the opposite option because its single win is either on a highly valid feature or involves a massive magnitude advantage.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design creates a perfect dissociation between WADD and Tallying. WADD strongly prefers Option A on trials 2, 3, 5, and 6, but strongly prefers Option B on trials 1 and 4. Tallying exhibits the opposite pattern: it prefers B on 3, 5, and 6, prefers A on 4, and ties on 1 and 2. By computing the difference in the proportion of A choices between the 'WADD-pro-A' trials and the 'WADD-anti-A' trials, we expect WADD to yield a score close to 1.0, while Tallying will yield a negative score (around -0.625). This large gap ensures robust discrimination between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert response to probability of choosing A (where response == 0)
    is_a = 1.0 - data['response']
    
    def get_trial_type(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (5, 5, 6, 6) and b == (10, 10, 0, 0): return 1
        if a == (0, 0, 8, 8) and b == (2, 2, 0, 0): return 2
        if a == (10, 0, 0, 0) and b == (0, 2, 2, 2): return 3
        if a == (1, 1, 1, 1) and b == (0, 0, 0, 10): return 4
        if a == (10, 0, 0, 0) and b == (7, 1, 1, 1): return 5
        if a == (0, 10, 0, 0) and b == (1, 0, 1, 1): return 6
        return 0
        
    tt = data.apply(get_trial_type, axis=1)
    
    # WADD predicts A on trials 2, 3, 5, 6 and B on trials 1, 4.
    # Tallying predicts B on 3, 5, 6, A on 4, and tie on 1, 2.
    wadd_pro_trials = is_a[tt.isin([2, 3, 5, 6])]
    wadd_anti_trials = is_a[tt.isin([1, 4])]
    
    wadd_pro = wadd_pro_trials.mean() if len(wadd_pro_trials) > 0 else 0.0
    wadd_anti = wadd_anti_trials.mean() if len(wadd_anti_trials) > 0 else 0.0
    
    return float(wadd_pro - wadd_anti)

### RESULTS
- Predicted under pi_3 (simulated): 0.3400 (var=0.1219)
- Predicted under pi_2 (simulated): -0.7500 (var=0.0000)
- Observed on real data: 0.3600 (var=0.1554)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**rating_max:** 10

**Trial pairs (n=8):**
  trial 1: A=[0, 3, 3, 3, 3]  B=[10, 0, 0, 0, 0]
  trial 2: A=[10, 0, 0, 0, 0]  B=[0, 3, 3, 3, 3]
  trial 3: A=[0, 0, 8, 8, 8]  B=[10, 10, 0, 0, 0]
  trial 4: A=[10, 10, 0, 0, 0]  B=[0, 0, 8, 8, 8]
  trial 5: A=[2, 2, 2, 2, 2]  B=[10, 10, 0, 0, 0]
  trial 6: A=[10, 10, 0, 0, 0]  B=[2, 2, 2, 2, 2]
  trial 7: A=[0, 0, 10, 10, 10]  B=[10, 10, 0, 0, 0]
  trial 8: A=[6, 6, 6, 6, 6]  B=[10, 0, 0, 0, 0]

**Rationale:** This design strictly dissociates Tallying from the Weighted Additive (WADD) rule by manipulating both the number of winning features and the cardinal magnitudes/validities of those features. Tallying ignores magnitude and validities, simply counting the number of features where one option is strictly greater than the other. WADD computes a validity-weighted sum of the cardinal values. The trials pit options that win on a majority of features but by small margins or on low-validity features against options that win on fewer features but by massive margins or on high-validity features. This creates stark preference reversals between the two theories.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Tallying heuristic's predictions on all trials where Tallying has a strict preference. The experimental design specifically pits Tallying against WADD by creating trials where one option wins on a majority of features (favored by Tallying) but the other option has a massive cardinal advantage on a few high-validity features (favored by WADD). Consequently, this metric will be close to 1.0 for agents using Tallying and much lower (often below 0.5) for agents using WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    tally_a = np.sum(a_mat > b_mat, axis=1)
    tally_b = np.sum(b_mat > a_mat, axis=1)
    
    valid_trials = tally_a != tally_b
    if not np.any(valid_trials):
        return 0.5
        
    tally_pred = (tally_b[valid_trials] > tally_a[valid_trials]).astype(int)
    actual_responses = data['response'].values[valid_trials]
    
    return float(np.mean(tally_pred == actual_responses))

### RESULTS
- Predicted under pi_3 (simulated): 0.5725 (var=0.0682)
- Predicted under pi_2 (simulated): 1.0000 (var=0.0000)
- Observed on real data: 0.5250 (var=0.0619)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[4, 2, 2, 2, 2]  B=[3, 4, 4, 4, 4]
  A=[2, 4, 4, 4, 4]  B=[4, 2, 2, 2, 2]
  A=[3, 4, 2, 2, 2]  B=[3, 3, 4, 4, 4]
  A=[5, 2, 4, 4, 4]  B=[5, 4, 2, 2, 2]
  A=[1, 1, 4, 2, 2]  B=[1, 1, 3, 4, 4]
  A=[2, 2, 2, 4, 4]  B=[2, 2, 4, 2, 2]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_pred = None
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                ttb_pred = 0
                break
            elif val_b > val_a:
                ttb_pred = 1
                break
        if ttb_pred is not None:
            matches.append(row['response'] == ttb_pred)
    if not matches:
        return 0.0
    return float(np.mean(matches))
```

**Observed (real) value:** 0.0733 (var=0.0124)
**Predicted under pi_3:** 0.0467 (var=0.0112)
**Predicted under pi_2:** 0.0000 (var=0.0000)

### Experiment 4
**Design**
  A=[5, 2, 2, 2]  B=[4, 4, 4, 4]
  A=[2, 5, 5, 5]  B=[4, 3, 3, 3]
  A=[5, 2, 3, 3]  B=[4, 4, 3, 3]
  A=[2, 5, 3, 3]  B=[4, 4, 3, 3]
  A=[3, 5, 2, 2]  B=[3, 4, 4, 4]
  A=[3, 2, 5, 5]  B=[3, 4, 4, 4]
  A=[3, 3, 5, 2]  B=[3, 3, 4, 4]
  A=[3, 3, 2, 5]  B=[3, 3, 4, 4]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_match = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins != b_wins:
            tally_pred = 0 if a_wins > b_wins else 1
            tally_match.append(1.0 if row['response'] == tally_pred else 0.0)
            
    if not tally_match:
        return 0.5
    return float(np.mean(tally_match))
```

**Observed (real) value:** 0.8350 (var=0.0340)
**Predicted under pi_3:** 0.8750 (var=0.0231)
**Predicted under pi_2:** 1.0000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (Weighted Additive, pi_3) consistently captures the observed data across all four experiments. Its point estimates closely match the observed values (e.g., in Experiment 1 it predicts 0.34 vs observed 0.36; in Experiment 2 it predicts 0.57 vs observed 0.525). Theory 2 (Tallying, pi_2) is excessively rigid, predicting extreme values (e.g., 1.0 or 0.0 or -0.75) that starkly fail to reflect human behavior. Because Theory 1 is performing exceptionally well and Theory 2 is failing completely, Theory 2 should be replaced with a brand-new theory that provides a more plausible competitor to the Weighted Additive rule.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose an 'Equal Weight' theory. Unlike Tallying (which only counts strict feature-wise wins) and Weighted Additive (which weights cardinal feature values by their validities), the Equal Weight heuristic assumes subjects sum the cardinal values of each option's features but ignore the cue validities, treating all features as equally important. This provides a strong, plausible alternative to WADD that accounts for magnitude differences without requiring complex validity weighting."
}
```

## Usage

```json
{
  "prompt_token_count": 5240,
  "candidates_token_count": 292,
  "total_token_count": 6364
}
```
