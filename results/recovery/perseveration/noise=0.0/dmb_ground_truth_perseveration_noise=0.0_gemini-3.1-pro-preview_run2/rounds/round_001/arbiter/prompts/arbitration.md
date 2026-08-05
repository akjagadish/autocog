# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_3
People use a Weighted Additive (WADD) strategy when comparing two options, but they subjectively scale the stated cue validities. By applying a non-linear transformation to the validities (exponentiating them by a parameter gamma), decision-makers can amplify the differences between cues (mimicking Take The Best) or compress them (mimicking Tallying). Expanding the range of gamma allows for extreme weight disparities, accommodating individuals who rely heavily on the most valid cue.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    gamma = float(parameters["gamma"])
    val = np.asarray(parameters["validities"], dtype=float) ** gamma
    
    # Compute the weighted sum of features for each option
    ev_a = np.sum(val * a)
    ev_b = np.sum(val * b)
    scores = np.array([ev_a, ev_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates Take The Best (TTB) from the generalized Weighted Additive (WADD) strategy by exploiting their different confidence (choice probability) predictions. TTB generates a binary internal score (1 for the winner, 0 for the loser) based solely on the first discriminating cue, completely ignoring the magnitude of the validities and the number of opposing cues. Consequently, TTB predicts a constant choice probability across all trials where a choice is made. By contrast, WADD computes a weighted sum. To mimic TTB's choices on trials where lower-validity cues outnumber the highest-validity cue (e.g., Trial 1), WADD requires a high 'gamma' parameter to heavily skew the weights. However, a high gamma causes the absolute difference in weighted sums to shrink exponentially for trials decided by lower-validity cues (e.g., Trial 4). Thus, WADD predicts highly variable choice probabilities across these trials (very confident when Cue 1 decides, near-guessing when Cue 4 decides), whereas TTB predicts uniform confidence.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
By design, Option A is always favored by the first discriminating cue in all 8 trial types. Therefore, Take The Best (TTB) predicts a constant probability of choosing Option A across all trials, depending only on the subject's noise parameters. Consequently, the standard deviation of the proportion of A choices across the 8 trial types should be very small for TTB (reflecting only binomial sampling noise). In contrast, the Weighted Additive (WADD) model evaluates the weighted sum of all cues. Because the cue trade-offs and absolute value differences vary drastically across the 8 trial types, WADD predicts highly variable choice probabilities for Option A (e.g., very confident on some trials, near-guessing or even preferring B on others, depending on the gamma parameter). Thus, WADD will yield a significantly higher standard deviation of choice proportions across trial types.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Create a hashable key for trial types based on Option A's features
    # (Since each of the 8 unique trial types has a distinct Option A, this uniquely identifies the trial type)
    trial_keys = data['option_a_ratings'].apply(tuple)
    
    # Calculate whether the subject chose Option A (response == 0)
    chose_A = (data['response'] == 0).astype(float)
    
    # Combine into a temporary DataFrame to group
    df_temp = pd.DataFrame({'trial_key': trial_keys, 'chose_A': chose_A})
    
    # Compute the proportion of Option A choices for each trial type
    props = df_temp.groupby('trial_key')['chose_A'].mean()
    
    # Return the standard deviation of these choice proportions across the 8 trial types
    return float(np.std(props))

### RESULTS
- Predicted under pi_1 (simulated): 0.0115 (var=0.0013)
- Predicted under pi_3 (simulated): 0.1069 (var=0.0046)
- Observed on real data: 0.0000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the generalized Weighted Additive (WADD) strategy from Take The Best (TTB) by exploiting their divergent predictions regarding choice confidence. TTB relies entirely on the first discriminating cue, ignoring both the absolute validity of that cue and any opposing lower-validity cues. Consequently, TTB predicts a constant choice probability (confidence) across all trials where a choice is made. Conversely, WADD computes a weighted sum of all cues, with validities transformed by a gamma parameter. Even if gamma is high (mimicking TTB's choices), the absolute difference in weighted sums shrinks exponentially for trials decided by lower-validity cues. Thus, WADD predicts highly variable choice probabilities across trials depending on which cue decides the choice and how many cues oppose it, while TTB predicts uniform confidence.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In the chosen experimental design, Take The Best (TTB) consistently identifies Option A as the winner across all 8 trial types because the first discriminating cue always favors Option A. Consequently, TTB predicts a uniform probability of choosing Option A across all trials, leading to a near-zero true variance in choice probabilities across trial types. In contrast, the Weighted Additive (WADD) strategy integrates all cues, meaning the number of opposing cues significantly modulates choice confidence. Thus, WADD predicts highly variable choice probabilities across the 8 trial types. By calculating the variance of each subject's mean response across the 8 unique trial pairs, this metric will yield a much higher value for data generated under WADD compared to TTB.

Source:
def metric(data: pd.DataFrame) -> float:
    # Create a hashable key for trial types based on option A (which is unique for all 8 pairs)
    keys = data['option_a_ratings'].apply(tuple)
    # Compute the mean response (proportion of B choices) per subject per trial type
    means = data.assign(trial_key=keys).groupby(['subject_id', 'trial_key'])['response'].mean()
    # Compute the variance of these means across the 8 trial types for each subject, then average
    return float(means.groupby('subject_id').var(ddof=0).mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.0089 (var=0.0000)
- Predicted under pi_3 (simulated): 0.0357 (var=0.0010)
- Observed on real data: 0.0000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    agreements = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_pred = -1
        # Features are in descending order of validity: [0.9, 0.8, 0.7, 0.6]
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if ttb_pred != -1:
            agreements.append(1.0 if resp == ttb_pred else 0.0)
            
    return float(np.mean(agreements)) if agreements else 0.0

```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_1:** 0.8390 (var=0.0119)
**Predicted under pi_3:** 0.6073 (var=0.0263)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    tallying_agreements = 0
    disagreement_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Tallying prediction
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            tally_pred = -1
            
        # Take The Best (TTB) prediction
        ttb_pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        # Check for strict opposition between the two heuristics
        if tally_pred != -1 and ttb_pred != -1 and tally_pred != ttb_pred:
            disagreement_trials += 1
            if row['response'] == tally_pred:
                tallying_agreements += 1
                
    if disagreement_trials == 0:
        return 0.5
    return tallying_agreements / disagreement_trials

```

**Observed (real) value:** 0.4400 (var=0.0242)
**Predicted under pi_1:** 0.1231 (var=0.0109)
**Predicted under pi_3:** 0.4475 (var=0.0386)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, the observed data strongly deviates from the deterministic predictions of Take The Best (THEORY 1). In Experiment 3, TTB agreement is exactly 0.5000 (chance level), and in Experiment 4, agreement with Tallying on disagreement trials is 0.4400, both of which suggest that subjects are either highly noisy or not utilizing the cue validities as instructed. In Experiments 1 and 2, the variance of choice proportions across trial types is exactly 0.0000, which is perfectly consistent with uniform random guessing (yielding exactly 50% A choices everywhere). THEORY 2 (WADD) gets closer to Experiments 3 and 4 by blending cue weights, but THEORY 1 (TTB) is far too rigid and overpredicts accuracy/agreement. Given that subjects appear to be guessing or ignoring validities entirely, TTB is fundamentally misaligned with the data.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Random Guessing / Baseline' theory or a 'Tallying' theory with extremely high noise. Since subjects show 0.5 agreement with TTB and 0.44 agreement with Tallying, they are largely guessing, likely due to the lack of feedback or low engagement. The new theory should explicitly model subjects as ignoring cue validities and making choices either uniformly at random or based on a simple unweighted tally of features with very high decision noise."
}
```

## Usage

```json
{
  "prompt_token_count": 5656,
  "candidates_token_count": 333,
  "total_token_count": 7229
}
```
