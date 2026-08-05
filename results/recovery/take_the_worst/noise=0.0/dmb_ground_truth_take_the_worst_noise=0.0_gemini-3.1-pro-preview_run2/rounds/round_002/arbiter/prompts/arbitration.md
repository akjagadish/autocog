# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Skeptical Tallying: Decision-makers primarily rely on a Tallying heuristic, counting the number of features where one option strictly dominates the other. However, when the tallies are tied (or closely matched), they do not simply guess. Instead, they exhibit skepticism toward the highest-validity cues—perhaps viewing them as redundant, overly salient, or manipulated—and systematically break ties by favoring options that possess more lower-validity features. This is modeled by augmenting the tally score with a secondary component that weights features inversely to their stated validity.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Skeptical Tallying expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    
    # Primary mechanism: Tallying (count of strict feature-wise wins)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Secondary mechanism: Tie-breaking favoring lower-validity features
    # Features are weighted by (1 - validity) so that lower validity cues provide a larger bonus.
    tie_breaker_a = np.sum(a * (1.0 - val))
    tie_breaker_b = np.sum(b * (1.0 - val))
    
    # Combine tally with the tie-breaker.
    # gamma controls the strength of the tie-breaker relative to a strict tally win.
    score_a = a_wins + gamma * tie_breaker_a
    score_b = b_wins + gamma * tie_breaker_b
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
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


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 1]

**Rationale:** This design systematically dissociates Standard Tallying from Skeptical Tallying by focusing on trials where the primary Tallying mechanism results in a tie. Under Standard Tallying, when both options win on an equal number of features, the model predicts a 50/50 guess regardless of the features' validities. In contrast, Skeptical Tallying breaks these ties by weighting the features inversely to their stated validity, favoring the option that possesses more lower-validity features. By constructing trials where option A wins on low-validity features and option B wins on high-validity features, Standard Tallying predicts indifference, whereas Skeptical Tallying predicts a clear preference for option A. Additional trials vary the magnitude of the tie-breaker difference to capture the effect across the parameter space.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
By design, every trial in this experiment features a perfect tie under Standard Tallying (both options win on the exact same number of features). Thus, Standard Tallying predicts that subjects will guess randomly, choosing Option A ~50% of the time. However, Option A is systematically constructed to win on lower-validity features, which Skeptical Tallying heavily weights as a tie-breaker. Therefore, Skeptical Tallying predicts a strong preference for Option A across all trials. The proportion of Option A choices will cleanly discriminate the two theories: ~0.5 for Standard Tallying, and >0.5 for Skeptical Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    # Return the proportion of trials where the subject chose Option A (response == 0)
    return float((data['response'] == 0).mean())


### RESULTS
- Predicted under pi_4 (simulated): 0.6873 (var=0.0154)
- Predicted under pi_2 (simulated): 0.4856 (var=0.0026)
- Observed on real data: 0.8817 (var=0.0046)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=7):** [0.95, 0.95, 0.95, 0.95, 0.5, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0, 1]

**Rationale:** To quantitatively dissociate Standard Tallying from Skeptical Tallying, we exploit the latter's 'tie-breaker' mechanism, which effectively functions as an additive bonus that scales inversely with feature validity. For Standard Tallying, all feature wins are weighted equally (weight = 1), and validities are ignored. For Skeptical Tallying, a feature win contributes 1 + gamma * (1 - validity). Thus, Skeptical Tallying assigns strictly higher weight to wins on low-validity features. We use 7 features (four high validity 0.95, three low validity 0.50). In 'tie' trials (e.g., A wins on 2 high-validity features, B wins on 2 low-validity features), Standard Tallying predicts a 50/50 guess, while Skeptical Tallying strongly prefers B. In 'opposition' trials (A wins 4 high-validity features, B wins 3 low-validity features), Standard Tallying deterministicly prefers A (4 wins vs 3), whereas Skeptical Tallying can flip its preference to B if the skepticism parameter gamma is sufficiently high, because the cumulative low-validity bonus outweighs the single extra win for A.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Tallying predicts that when the number of positive features is equal for both options, the decision-maker will guess, resulting in a 50% probability of choosing option B. Skeptical Tallying, however, breaks these ties by favoring options with lower-validity features. Trials 1, 2, 3, and 5 are designed precisely as such 'tie' trials, where option B consists of lower-validity features. The metric computes the proportion of option B choices on these tie trials. For Standard Tallying, this proportion will be centered at 0.5. For Skeptical Tallying, this proportion will be systematically greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    # Calculate sum of ratings for A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Filter for tie trials where both options have the same number of positive features
    tie_trials = data[sum_a == sum_b]
    
    if len(tie_trials) == 0:
        return 0.5
        
    return float(tie_trials['response'].mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.7891 (var=0.0184)
- Predicted under pi_2 (simulated): 0.4988 (var=0.0035)
- Observed on real data: 0.6781 (var=0.0038)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_ttb_pred(a, b):
        for val_a, val_b in zip(a, b):
            if val_a > val_b:
                return 0
            if val_b > val_a:
                return 1
        return -1

    preds = [get_ttb_pred(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    
    matches = sum(1 for p, r in zip(preds, data['response']) if p == r)
    valid = sum(1 for p in preds if p != -1)
    
    return float(matches) / valid if valid > 0 else 0.0
```

**Observed (real) value:** 0.1564 (var=0.0079)
**Predicted under pi_4:** 0.1758 (var=0.0092)
**Predicted under pi_2:** 0.2378 (var=0.0054)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    tally_preds = np.where(a_wins > b_wins, 0, 1)
    actual_responses = data['response'].values
    
    matches = (tally_preds[mask] == actual_responses[mask])
    return float(np.mean(matches))
```

**Observed (real) value:** 0.8943 (var=0.0064)
**Predicted under pi_4:** 0.8652 (var=0.0101)
**Predicted under pi_2:** 0.8726 (var=0.0064)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    
    wadd_aligned = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        
        # Focus only on "tie" trials where Tallying predicts 50/50
        if sum_a == sum_b:
            val_a = np.sum(a * validities)
            val_b = np.sum(b * validities)
            
            if val_a > val_b:
                wadd_pref = 0
            elif val_b > val_a:
                wadd_pref = 1
            else:
                continue
                
            wadd_aligned.append(1 if row['response'] == wadd_pref else 0)
            
    if not wadd_aligned:
        return 0.5
        
    return float(np.mean(wadd_aligned))
```

**Observed (real) value:** 0.1450 (var=0.0093)
**Predicted under pi_4:** 0.3325 (var=0.0202)
**Predicted under pi_2:** 0.4875 (var=0.0173)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.99, 0.95, 0.55, 0.52, 0.5])
    
    wadd_acc = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        score_a = np.sum(val * a)
        score_b = np.sum(val * b)
        
        wadd_prefers_a = (score_a > score_b)
        wadd_prefers_b = (score_b > score_a)
        
        tally_prefers_a = (a_wins > b_wins)
        tally_prefers_b = (b_wins > a_wins)
        
        # Filter for trials where Tallying does NOT agree with WADD
        # (i.e. Tallying is tied, or Tallying actively prefers the opposite)
        if (wadd_prefers_a and not tally_prefers_a) or \
           (wadd_prefers_b and not tally_prefers_b):
            
            if wadd_prefers_a:
                wadd_correct = (row['response'] == 0)
            else:
                wadd_correct = (row['response'] == 1)
                
            wadd_acc.append(float(wadd_correct))
            
    if len(wadd_acc) == 0:
        return 0.5
    return float(np.mean(wadd_acc))
```

**Observed (real) value:** 0.1250 (var=0.0083)
**Predicted under pi_4:** 0.2131 (var=0.0166)
**Predicted under pi_2:** 0.3792 (var=0.0038)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Skeptical Tallying) vastly outperforms Theory 2 (Standard Tallying). The data show a striking and consistent pattern: when the standard tally of features results in a tie (or near-tie), subjects do not simply guess. Instead, they systematically and strongly prefer the option that wins on the *lowest* validity features. This is evident in Exp 1 (observed 0.88 for the low-validity option vs. Theory 2's predicted 0.48), Exp 2 (observed 0.67 vs. predicted 0.49), Exp 5 (WADD alignment on ties is 0.14, meaning subjects reliably anti-align with high-validity cues), and Exp 6. While Theory 1 captures this qualitative 'reactance' or skepticism perfectly, Theory 2's assumption of uniform random guessing during ties is cleanly falsified. Because Theory 1 is already so dominant, Theory 2 should be completely replaced with a new, stronger competitor that also attempts to explain this low-validity preference.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory to replace Standard Tallying. The new theory should be 'Low-Validity Lexicographic' or 'Anti-Expertise WADD'. Instead of using Tallying as a primary mechanism and low-validity as a tie-breaker (like Theory 1), this new theory should posit that subjects actively distrust high-validity 'expert' cues entirely. They might completely discard features with validities above a certain threshold (e.g., > 0.75) and only perform Tallying or WADD on the remaining low-validity features. This provides a structurally distinct mechanism from Theory 1 (filtering/attention rather than tie-breaking) while still predicting the strong preference for low-validity features observed in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 6867,
  "candidates_token_count": 429,
  "total_token_count": 8932
}
```
