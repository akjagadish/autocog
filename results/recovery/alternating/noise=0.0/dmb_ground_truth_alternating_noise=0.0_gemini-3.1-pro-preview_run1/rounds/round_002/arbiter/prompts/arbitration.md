# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Zero-Intelligence / Random Choice Theory: In this specific experimental paradigm, participants do not systematically integrate feature validities or use non-compensatory heuristics. Due to task disengagement, cognitive overload, or the lack of trial-by-trial feedback, participants do not learn or apply any complex decision rules. Instead, they simply guess randomly between the two options on every trial, resulting in a 50/50 probability for any pair of options regardless of their features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The Zero-Intelligence model completely ignores the state and history,
    # assuming participants guess uniformly at random on every trial.
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_3
Weighted Additive (WADD) Theory: Decision-makers evaluate options by computing a weighted sum of their features, where the weights directly correspond to the cue validities. The probability of choosing an option is then determined by the difference in these weighted sums via a softmax function, subject to decision noise. To capture empirical data where choices appear highly noisy or evenly balanced across experimental manipulations, the decision noise parameters (softmax inverse temperature and random lapse rate) can span the full range from deterministic choice to complete uniform guessing. In this iteration, the inverse temperature is constrained to very low values, enforcing highly stochastic behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) state.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute the weighted sum of features for each option
    wadd_a = np.sum(stim[0] * val)
    wadd_b = np.sum(stim[1] * val)
    
    scores = np.array([wadd_a, wadd_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 6: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** This design tests the Zero-Intelligence (Random Choice) theory against the heavily stochastic Weighted Additive (WADD) theory. By parametrically varying the difference in the weighted sums of features between Option A and Option B from extreme to near-zero, WADD predicts a graded, sigmoid-like shift in choice probabilities (even with its highly constrained, low beta). In contrast, the Zero-Intelligence theory predicts a completely flat choice probability of 50% across all trials, regardless of how overwhelmingly one option dominates the other in feature value. A lack of sensitivity to these massive feature differences would strongly dissociate Zero-Intelligence from WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric computes the proportion of choices that align with the Weighted Additive (WADD) theory's deterministically preferred option (i.e., the option with the higher weighted sum of features). For the Zero-Intelligence theory, participants guess randomly, so this metric will tightly center around 0.5. For the WADD theory, even with high stochasticity (low beta or high epsilon), the metric will systematically deviate above 0.5 because the choices are still fundamentally driven by the weighted feature differences. Computing the mean over all trials ensures low within-subject and between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    def get_wadd_diff(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * val) - np.sum(b * val)
        
    diffs = data.apply(get_wadd_diff, axis=1)
    
    # WADD predicts choice A (0) when diffs > 0, and choice B (1) when diffs < 0
    is_correct = ((diffs > 0) & (data['response'] == 0)) | ((diffs < 0) & (data['response'] == 1))
    
    return float(is_correct.mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.4856 (var=0.0028)
- Predicted under pi_3 (simulated): 0.5494 (var=0.0039)
- Observed on real data: 0.5008 (var=0.0004)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 2: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 4: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 5: A=[0, 1, 1, 0, 0, 1]  B=[1, 0, 0, 1, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]
  trial 7: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]

**Rationale:** This design tests the heavily stochastic Weighted Additive (WADD) theory against the Zero-Intelligence (Random Choice) theory by systematically varying the difference in weighted sums between Option A and Option B across a wide range. While WADD is constrained to low beta values (implying high stochasticity), it still predicts that choice probabilities will monotonically track the difference in weighted feature sums. In contrast, the Zero-Intelligence theory predicts a completely flat 50/50 choice probability across all pairs regardless of how heavily one option dominates the other. By including pairs with extreme differences as well as very subtle differences, we can observe whether choices exhibit the graded, sigmoid-like sensitivity predicted by WADD or the complete insensitivity predicted by Random Choice.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The previous unweighted accuracy metric failed because it treated all trials equally, diluting the signal with pairs where the options were practically identical in value (where WADD and ZI both predict ~50/50). By weighting the accuracy on each trial by the absolute difference in the options' weighted sums, this metric gives much more leverage to the 'extreme' pairs (trials 1 and 8) where WADD predicts a much stronger preference than 50/50, while gracefully downweighting the ambiguous pairs. This amplifies the mean gap between the theories while maintaining low variance by still incorporating all data.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    
    A = np.array(data['option_a_ratings'].tolist())
    B = np.array(data['option_b_ratings'].tolist())
    
    val_a = A.dot(val)
    val_b = B.dot(val)
    
    # The expected choice under deterministic WADD
    expected = (val_a < val_b).astype(int)
    correct = (data['response'].values == expected).astype(float)
    
    # Weight the accuracy by the absolute difference in weighted sums
    diff = np.abs(val_a - val_b)
    
    # Return the weighted accuracy
    return float(np.sum(correct * diff) / np.sum(diff))

### RESULTS
- Predicted under pi_4 (simulated): 0.4894 (var=0.0064)
- Predicted under pi_3 (simulated): 0.5956 (var=0.0133)
- Observed on real data: 0.4990 (var=0.0006)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # TTB choice: first cue that discriminates determines choice
        ttb_choice = None
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_choice = 0
                break
            elif b[j] > a[j]:
                ttb_choice = 1
                break
                
        if ttb_choice is None:
            ttb_choice = 0.5
            
        if ttb_choice == row['response']:
            matches.append(1.0)
        elif ttb_choice == 0.5:
            matches.append(0.5)
        else:
            matches.append(0.0)
            
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.5046 (var=0.0028)
**Predicted under pi_3:** 0.4742 (var=0.0035)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = 0
    valid_trials = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_arr = np.array(a)
        b_arr = np.array(b)
        a_wins = np.sum(a_arr > b_arr)
        b_wins = np.sum(b_arr > a_arr)
        if a_wins > b_wins:
            if r == 0:
                matches += 1
            valid_trials += 1
        elif b_wins > a_wins:
            if r == 1:
                matches += 1
            valid_trials += 1
    return float(matches / valid_trials) if valid_trials > 0 else 0.5
```

**Observed (real) value:** 0.5042 (var=0.0108)
**Predicted under pi_4:** 0.4954 (var=0.0020)
**Predicted under pi_3:** 0.5233 (var=0.0024)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.5029 (var=0.0019)
**Predicted under pi_3:** 0.5192 (var=0.0039)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4985 (var=0.0000)
**Predicted under pi_4:** 0.5069 (var=0.0030)
**Predicted under pi_3:** 0.5147 (var=0.0036)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, the observed data consistently yields metric values of approximately 0.500 (with near-zero variance). THEORY 1 (Zero-Intelligence / Random Choice) perfectly captures this flatline behavior, predicting ~0.5 on every metric. THEORY 2 (Weighted Additive), despite being constrained to high stochasticity, consistently over-predicts the degree to which participants' choices are sensitive to the feature validities (e.g., predicting 0.595 vs observed 0.499 in Experiment 2). The empirical evidence overwhelmingly supports the hypothesis that participants are guessing randomly, making WADD fundamentally mismatched to the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "THEORY 1 (Zero-Intelligence) is clearly the superior model. To provide a stronger, more plausible competitor than WADD, propose a new theory based on an extremely simplified heuristic, such as 'Tallying with High Lapse' (participants simply count the number of positive features but are subject to near-total decision noise) or a 'Single-Cue / First-Feature' heuristic where only the most valid cue is ever considered. This will test whether there is any residual, highly simplified cognitive strategy at play rather than the computationally expensive weighted integration assumed by WADD."
}
```

## Usage

```json
{
  "prompt_token_count": 5513,
  "candidates_token_count": 291,
  "total_token_count": 6637
}
```
