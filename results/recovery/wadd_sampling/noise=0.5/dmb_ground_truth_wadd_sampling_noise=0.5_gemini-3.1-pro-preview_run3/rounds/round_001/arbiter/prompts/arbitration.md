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
Weighted Additive (WADD) Theory with Log-Odds: People compare two options by considering all available features, weighting each feature by its subjective validity transformed into log-odds (Bayesian weights), and summing these weighted values to form an overall valuation for each option. This normative integration allows multiple lower-validity cues to jointly outweigh a single high-validity cue, while the non-linear log-odds transformation ensures that the highest-validity cues maintain sufficient relative influence. Response noise is modeled via a softmax over the WADD scores and an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Convert validities to log-odds weights for Bayesian integration.
    # Clip validities to avoid division by zero or log(0) for extreme values.
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    a, b = stim[0], stim[1]
    
    # Calculate Weighted Additive scores for both options
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


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
**Validities (n_features=4):** [0.95, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 3: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 5: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 8: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** This design pits Weighted Additive (WADD) Theory with Log-Odds against Tallying. WADD integrates features by weighting them with the log-odds of their validities, allowing a single highly valid cue to outweigh multiple less valid cues, or differentiating between options when the number of winning cues is tied. Tallying simply counts the number of features each option wins on, ignoring validities. By using validities [0.95, 0.75, 0.65, 0.55], we create strong disparities between log-odds weights. We include trials where one option wins on a single highly valid cue while the other wins on three less valid cues (WADD favors the former, Tallying favors the latter), as well as trials where both options win on an equal number of cues but with different validities (WADD strongly prefers one option, Tallying predicts a tie).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the WADD (Weighted Additive) theory on the specific subset of trials where WADD and Tallying make strictly opposing predictions. For example, when option A has only the single most valid cue and option B has the three less valid cues, WADD strongly prefers A while Tallying strongly prefers B. By isolating these trials, the metric will be close to 1 under WADD and close to 0 under Tallying, providing maximum discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total_relevant = 0
    
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 1: WADD prefers A, Tallying prefers B
        if a == [1, 0, 0, 0] and b == [0, 1, 1, 1]:
            wadd_choices += (1 if resp == 0 else 0)
            total_relevant += 1
        # Trial 3: WADD prefers A, Tallying prefers B
        elif a == [0, 1, 0, 0] and b == [0, 0, 1, 1]:
            wadd_choices += (1 if resp == 0 else 0)
            total_relevant += 1
        # Trial 6: WADD prefers B, Tallying prefers A
        elif a == [0, 0, 1, 1] and b == [0, 1, 0, 0]:
            wadd_choices += (1 if resp == 1 else 0)
            total_relevant += 1
        # Trial 8: WADD prefers B, Tallying prefers A
        elif a == [0, 1, 1, 1] and b == [1, 0, 0, 0]:
            wadd_choices += (1 if resp == 1 else 0)
            total_relevant += 1
            
    if total_relevant == 0:
        return 0.5
    return wadd_choices / total_relevant

### RESULTS
- Predicted under pi_3 (simulated): 0.8171 (var=0.0142)
- Predicted under pi_2 (simulated): 0.1437 (var=0.0072)
- Observed on real data: 0.3858 (var=0.0151)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.6, 0.55, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** To strongly dissociate Tallying (which simply counts the number of features each option wins on, ignoring validities) from Weighted Additive (WADD) with log-odds (which scales features non-linearly by their validities), we use a set of five features with a steep drop-off in validity (0.90, 0.80, 0.60, 0.55, 0.55). This ensures that the log-odds weight of the first feature is substantially larger than the sum of the bottom three or four features. In these trials, Option A often wins on a sheer majority of low-validity features (strongly favored by Tallying), while Option B wins on only one or two highly valid features (strongly favored by WADD). We also include trials where both options win on an equal number of features (Tallying predicts a tie), but the validities heavily favor one option (WADD strongly predicts that option).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design was specifically crafted such that Tallying (which counts raw feature wins) and WADD (which weights features by log-odds validities) strictly disagree on almost all trials. For example, Option A might win on 3 low-validity features while Option B wins on 1 high-validity feature. Tallying will consistently prefer Option A, whereas WADD will prefer Option B. This metric calculates the proportion of trials where the subject's choice aligns with the Tallying heuristic's strict preference. Data simulated under Tallying will yield a high proportion (approaching 1.0 for low noise), whereas data simulated under WADD will yield a very low proportion (approaching 0.0), strongly discriminating the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    consistent = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            if row['response'] == 0:
                consistent += 1
            total += 1
        elif b_wins > a_wins:
            if row['response'] == 1:
                consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return consistent / total

### RESULTS
- Predicted under pi_3 (simulated): 0.2057 (var=0.0076)
- Predicted under pi_2 (simulated): 0.8564 (var=0.0067)
- Observed on real data: 0.5610 (var=0.0074)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    diff = a_mat - b_mat
    non_zero = diff != 0
    
    first_diff_idx = np.argmax(non_zero, axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    
    ttb_choice = np.where(first_diff_val > 0, 0, 1)
    has_diff = np.any(non_zero, axis=1)
    
    if not np.any(has_diff):
        return 0.5
        
    match = (data['response'].values[has_diff] == ttb_choice[has_diff])
    return float(np.mean(match))
```

**Observed (real) value:** 0.4083 (var=0.0075)
**Predicted under pi_3:** 0.3802 (var=0.0046)
**Predicted under pi_2:** 0.1250 (var=0.0061)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = (a_mat > b_mat).sum(axis=1)
    b_wins = (b_mat > a_mat).sum(axis=1)
    
    preds = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    valid = preds != -1
    if not valid.any():
        return 0.5
        
    matches = (data['response'].values[valid] == preds[valid])
    return float(matches.mean())
```

**Observed (real) value:** 0.6125 (var=0.0092)
**Predicted under pi_3:** 0.5902 (var=0.0031)
**Predicted under pi_2:** 0.8490 (var=0.0088)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither theory perfectly captures the observed data. Theory 1 (WADD with Log-Odds) predicts extreme choices in favor of the highest-validity cues (e.g., Exp 1 predicted 0.81, observed 0.38), because the log-odds transformation creates massive disparities between weights. Theory 2 (Tallying) predicts extreme choices in favor of cue count (e.g., Exp 2 predicted 0.85, observed 0.56) and completely fails on Experiment 3 (predicted 0.12, observed 0.40). The observed data consistently falls between the extremes of pure Tallying and Log-Odds WADD, suggesting subjects are weighting cues but not as aggressively as Bayesian log-odds would prescribe.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace Tallying with a 'Linear Weighted Additive' theory. In this theory, subjects compute an overall score for each option by weighting each feature directly by its raw validity (or validity minus 0.5, representing its predictive edge over chance), rather than using a non-linear log-odds transformation. This provides a moderate integration strategy that captures the middle ground between ignoring validities (Tallying) and over-weighting the top validities (Log-Odds WADD), which should better match the intermediate choice proportions observed across all experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 5509,
  "candidates_token_count": 317,
  "total_token_count": 6768
}
```
