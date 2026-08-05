# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Weighted Additive (WADD) Model: Decision-makers evaluate options by computing a compensatory score for each, weighting every feature by its subjective importance (the log-odds of its validity). To balance extreme validities, decision-makers apply a smoothing or regularization process (Laplace smoothing) to the validities before computing their log-odds. This allows multiple weaker cues to perfectly balance a single strong cue, naturally yielding near-tied evaluations for adversarial choice pairs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    kappa = float(parameters["kappa"])
    
    # Laplace smoothing to regularize extreme validities
    val_smoothed = (val + kappa) / (1.0 + 2.0 * kappa)
    
    # Transform smoothed validities to log-odds for normative compensatory weighting
    # Clip to avoid log(0) or division by zero
    val_clipped = np.clip(val_smoothed, 0.001, 0.999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute WADD scores as the dot product of features and weights
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_4
Overload/Indifference Theory: When faced with adversarial choices where one highly valid cue conflicts with multiple weakly valid cues, subjects experience cognitive overload or perceive the options as perfectly tied. This leads to a breakdown of compensatory or non-compensatory decision strategies, resulting in pure random guessing (a 50/50 choice probability on every trial).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    # Read dummy parameter to satisfy the parameter reference contract
    _ = float(parameters["dummy"])
    
    stim = np.asarray(state, dtype=float)
    n_opts = stim.shape[0]
    
    # Pure random guessing due to cognitive overload / indifference
    return np.ones(n_opts) / n_opts


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To cleanly dissociate the Weighted Additive (WADD) Model from Overload/Indifference Theory, this design pits a single highly predictive cue against varying numbers of moderately or weakly predictive cues. Overload/Indifference Theory posits that such adversarial conflicts cause cognitive overload, resulting in pure random guessing (a 50/50 choice probability) across all these trials. Conversely, WADD leverages Laplace smoothing to regularize the extreme validities and then computes a compensatory log-odds sum. Under WADD, these conflicts do not yield perfect ties; instead, the model predicts systematic, non-random preferences depending on the exact balance of the smoothed log-odds weights. By varying the number and strength of the opposing weak cues, we generate trials where WADD predicts distinct, directional choices (away from 0.5), sharply contrasting with the universal 0.5 prediction of the Overload theory.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Overload theory predicts pure random guessing on all trials, meaning the choice proportion for any trial type should be exactly 0.5 in expectation. Given 12 repetitions per trial type, the mean absolute deviation from 0.5 under Overload will be strictly governed by binomial noise (expected value ~0.113). In contrast, WADD predicts systematic deviations from 0.5 due to compensatory weighting of the smoothed validities, leading to choice proportions closer to 0 or 1. By computing the mean absolute deviation of choice proportions from 0.5 across unique trial types, we can cleanly separate the random guessing of Overload from the deterministic preferences of WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    df['trial_type'] = df['option_a_ratings'].apply(tuple)
    trial_means = df.groupby(['subject_id', 'trial_type'])['response'].mean()
    subj_devs = (trial_means - 0.5).abs().groupby('subject_id').mean()
    return float(subj_devs.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.1671 (var=0.0056)
- Predicted under pi_4 (simulated): 0.1110 (var=0.0010)
- Observed on real data: 0.5000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 10: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To conclusively demonstrate Overload/Indifference Theory over the Weighted Additive (WADD) model, we must prevent WADD from using a global 'guessing' parameter (like epsilon=1.0 or beta=0.0) to fit the data. We achieve this by mixing 'strict dominance' trials (where one option is clearly superior and subjects should rarely guess) with 'adversarial conflict' trials. In the conflict trials, a single highly predictive cue is pitted against varying numbers of weakly predictive cues (1 to 4 weak cues). WADD computes a compensatory sum based on smoothed log-odds; because the number of weak cues varies, WADD mathematically must predict a shifting preference across these conflict trials. Overload/Indifference Theory, however, predicts a flat 50/50 guessing rate across all adversarial conflicts due to cognitive overload, regardless of the exact number of weak cues. If subjects show strong preferences on dominance trials but flat 50/50 choices on all conflict trials, WADD will fail to fit the data.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Overload/Indifference Theory predicts random guessing (0.5) across all trials, completely ignoring cue values. WADD, on the other hand, evaluates the compensatory score of each option. On pure dominance trials (where one option has positive cues and the other has none, such as trials 1-4), WADD will strongly prefer the dominant option, while Overload/Indifference Theory will still predict a 50% choice rate. This metric calculates the proportion of dominant choices on these clear trials, ensuring a large and highly significant divergence between the two models.

Source:
def metric(data: pd.DataFrame) -> float:
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    dom_a = (a_sum > 0) & (b_sum == 0)
    dom_b = (a_sum == 0) & (b_sum > 0)
    subset = data[dom_a | dom_b]
    if len(subset) == 0:
        return 0.5
    correct = ((dom_a) & (data['response'] == 0)) | ((dom_b) & (data['response'] == 1))
    return float(correct[dom_a | dom_b].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.6669 (var=0.0240)
- Predicted under pi_4 (simulated): 0.5106 (var=0.0098)
- Observed on real data: 0.5000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    ttb_match = 0
    total = 0
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        # TTB prediction (cues ordered 0 to 4 based on validities)
        ttb_winner = None
        for j in range(5):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying prediction
        a_wins = sum(1 for i in range(5) if a[i] > b[i])
        b_wins = sum(1 for i in range(5) if b[i] > a[i])
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        # Only evaluate on trials where the two heuristics strictly oppose
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if resp == ttb_winner:
                ttb_match += 1
            total += 1
            
    return float(ttb_match / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.4317 (var=0.0054)
**Predicted under pi_4:** 0.4833 (var=0.0028)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = (a_ratings > b_ratings).sum(axis=1)
    b_wins = (b_ratings > a_ratings).sum(axis=1)
    
    tally_pred = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    matches = (data['response'] == tally_pred)
    return float(matches.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.5667 (var=0.0047)
**Predicted under pi_4:** 0.5021 (var=0.0022)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    b_has_f1 = data['option_b_ratings'].apply(lambda x: x[0] == 1).astype(int)
    chose_f1 = (data['response'] == b_has_f1)
    return float(chose_f1.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.6669 (var=0.0149)
**Predicted under pi_4:** 0.4900 (var=0.0024)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])
    chose_f1 = ((a_f1 == 1) & (data['response'] == 0)) | ((b_f1 == 1) & (data['response'] == 1))
    return float(chose_f1.mean())
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_3:** 0.6677 (var=0.0158)
**Predicted under pi_4:** 0.4919 (var=0.0026)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metric is exactly 0.5000 with a variance of 0.0000. In Experiment 1, the metric is the mean absolute deviation of choice proportions from 0.5. A value of exactly 0.5 means that subjects' choice proportions for any given trial type were perfectly deterministic (either 1.0 or 0.0), completely contradicting Theory 2's prediction of trial-by-trial random guessing (which would yield a deviation of ~0.11 due to binomial noise). Furthermore, in Experiment 2 (and 3-6), the metrics test accuracy or heuristic-matching on balanced trial sets. An exact 0.5 score with 0 variance across all these balanced designs mathematically implies that subjects are deterministically choosing the exact same option position (e.g., always Option A) regardless of the features. Neither a compensatory model (Theory 1) nor a random guessing model (Theory 2) captures this purely deterministic, feature-ignoring position bias.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Position Bias' or 'Task Disengagement' theory. The data perfectly aligns with subjects entirely ignoring the cue validities and features, instead relying on a deterministic heuristic of simply selecting the first option (Option A) on every single trial. This explains the 0.5 absolute deviation from 0.5 in Exp 1 (since proportions are 1.0 or 0.0) and exactly 50% match rates on all other experiments where trial types are symmetrically balanced between Option A and Option B. The new model should predict a 1.0 probability for Option A (or a strong deterministic position bias parameter)."
}
```

## Usage

```json
{
  "prompt_token_count": 5941,
  "candidates_token_count": 386,
  "total_token_count": 8110
}
```
