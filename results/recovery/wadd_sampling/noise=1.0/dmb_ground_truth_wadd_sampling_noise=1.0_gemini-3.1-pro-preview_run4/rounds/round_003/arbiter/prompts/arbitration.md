# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Due to the abstract nature of the task and lack of trial-by-trial feedback, subjects do not consistently apply structured multi-attribute heuristics. Instead, their choices are dominated by high response noise or a massive lapse rate, effectively resulting in random guessing on most trials. While they may occasionally attempt to tally features or look at the most valid cue, the lack of grounding leads to a near-uniform choice probability across all stimulus pairs.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    
    # Calculate a weak tallying signal
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Massive lapse rate dominates the choice
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_5
Faced with abstract options and no trial-by-trial feedback, subjects abandon feature-based compensatory or non-compensatory strategies. Instead, they fall back on a strong, idiosyncratic side bias (e.g., a spatial preference for the left or right option, or a sequential preference for the first or second option). Each subject has a fixed probability of choosing Option A, which varies uniformly across the population from 0 to 1. This mechanism produces behavior that is completely independent of the feature validities or values, resulting in an aggregate alignment of ~50% with any feature-based heuristic (like Take The Best or Tallying), but driven by individual-level spatial/sequential biases rather than trial-level random guessing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    # The subject ignores the stimulus features and relies entirely on their idiosyncratic side bias.
    p_a = float(parameters["p_a"])
    
    # Return the fixed choice probabilities for Option A and Option B
    return np.array([p_a, 1.0 - p_a])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 8: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To dissociate the Advocated Theory (trial-by-trial random guessing due to massive lapse rates) from the Competing Theory (fixed idiosyncratic side bias), we present a diverse set of trials ranging from overwhelming evidence for one option to highly conflicting cue profiles. Both theories predict that subjects will ignore the feature evidence. However, they differ drastically in their predictions for within-subject behavior: the Advocated Theory predicts that every subject's choice proportion for Option A will tightly cluster around 0.5 (reflecting trial-by-trial high-noise guessing), whereas the Competing Theory predicts a wide, uniform distribution of individual choice proportions across the population (reflecting strong, fixed individual side biases, where some subjects always choose A and others always choose B).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The Advocated Theory predicts that subjects guess randomly on almost every trial due to high noise/lapse rates, so their individual choice proportions will tightly cluster around 0.5. The absolute deviation from 0.5 will be close to 0. In contrast, the Competing Theory predicts idiosyncratic, fixed side biases uniformly distributed between 0 and 1 across the population. The absolute deviation from 0.5 for these subjects will be uniformly distributed between 0 and 0.5, yielding a much higher population mean (~0.25) and distinct variance. Averaging the subject-level absolute deviations cleanly separates the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the proportion of 'B' choices (response == 1) for each subject
    # Then compute the absolute deviation from 0.5 (random guessing)
    # Average this deviation across all subjects in the data
    return float(data.groupby('subject_id')['response'].mean().apply(lambda x: abs(x - 0.5)).mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.0383 (var=0.0007)
- Predicted under pi_5 (simulated): 0.2338 (var=0.0199)
- Observed on real data: 0.0479 (var=0.0010)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** The Advocated Theory posits a fixed idiosyncratic side bias, predicting that a subject's probability of choosing Option A remains exactly constant across all trials, completely ignoring feature evidence. The Competing Theory posits high response noise but retains a weak sensitivity to the tally of winning features (due to epsilon < 1.0 and beta > 0). By presenting pairs with extreme tally differences (e.g., 5-0 vs 0-5), the Competing Theory predicts a measurable shift in the choice probability for A across these conditions (e.g., higher when A has 5 winning features than when B has 5). In contrast, the Advocated Theory predicts zero shift in choice probability across these conditions for any given subject, maintaining a flat line across all tally differences.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
The Advocated Theory posits that subjects rely on a fixed, idiosyncratic side bias, meaning each subject's overall proportion of choosing Option A will be uniformly distributed between 0 and 1. This results in a large average absolute deviation from 0.5 (expected value ~0.25). The Competing Theory posits that subjects make noisy, near-random choices on every trial, meaning every subject's overall proportion of choosing Option A will tightly cluster around 0.5 (expected value ~0.04 for 96 trials). Measuring the average absolute deviation of each subject's choice proportion from 0.5 effectively discriminates the population-level variance of side biases (Advocated) from trial-level response noise (Competing).

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    subject_means = data.groupby('subject_id')['response'].mean()
    return float(np.abs(subject_means - 0.5).mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.0338 (var=0.0008)
- Predicted under pi_5 (simulated): 0.2696 (var=0.0189)
- Observed on real data: 0.0392 (var=0.0007)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract matrices of option features
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Calculate the difference
    diff = a_mat - b_mat
    
    # Using exponentially decreasing weights, the sign of the weighted sum
    # is completely determined by the first non-zero difference.
    # This exactly mimics the one-reason stopping rule of Take The Best.
    weights = np.array([10000, 1000, 100, 10, 1])
    weighted_diff = np.dot(diff, weights)
    
    # If weighted_diff > 0, A is favored by the first discriminating cue (TTB predicts 0).
    # If weighted_diff < 0, B is favored by the first discriminating cue (TTB predicts 1).
    ttb_pred = (weighted_diff < 0).astype(int)
    
    # Return the proportion of choices that align with the Take The Best prediction
    return float(np.mean(ttb_pred == data['response']))
```

**Observed (real) value:** 0.5025 (var=0.0034)
**Predicted under pi_4:** 0.4881 (var=0.0020)
**Predicted under pi_5:** 0.5038 (var=0.0059)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        tally_pred = 0 if a_wins > b_wins else 1
        matches.append(row['response'] == tally_pred)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5121 (var=0.0032)
**Predicted under pi_4:** 0.5106 (var=0.0028)
**Predicted under pi_5:** 0.4965 (var=0.0015)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 1]  B=[1, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        # Focus only on trials where Tallying predicts a tie
        if tally_a == tally_b:
            ttb_winner = None
            # Cue order matches index order since validities are strictly decreasing
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_winner = 0
                    break
                elif b[i] > a[i]:
                    ttb_winner = 1
                    break
            
            if ttb_winner is not None:
                matches.append(row['response'] == ttb_winner)
                
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5117 (var=0.0085)
**Predicted under pi_4:** 0.4883 (var=0.0066)
**Predicted under pi_5:** 0.5175 (var=0.0068)

### Experiment 6
**Design**
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_aligned = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            for i in range(len(a)):
                if a[i] > b[i]:
                    ttb_choice = 0
                    break
                elif b[i] > a[i]:
                    ttb_choice = 1
                    break
            else:
                continue
                
            ttb_aligned.append(int(row['response'] == ttb_choice))
            
    if len(ttb_aligned) == 0:
        return 0.5
    return float(np.mean(ttb_aligned))
```

**Observed (real) value:** 0.4842 (var=0.0049)
**Predicted under pi_4:** 0.4808 (var=0.0053)
**Predicted under pi_5:** 0.4958 (var=0.0046)

### Experiment 7
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tally_choice = np.where(a_wins > b_wins, 0, 1)
    valid = a_wins != b_wins
    
    responses = data['response'].values
    match = (responses[valid] == tally_choice[valid])
    
    return float(np.mean(match)) if len(match) > 0 else 0.5

```

**Observed (real) value:** 0.5021 (var=0.0041)
**Predicted under pi_4:** 0.5262 (var=0.0019)
**Predicted under pi_5:** 0.4908 (var=0.0021)

### Experiment 8
**Design**
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    responses = data['response'].values
    
    valid_mask = a_wins != b_wins
    if not np.any(valid_mask):
        return 0.5
        
    tally_winner = (b_wins > a_wins).astype(int)
    matches = (responses[valid_mask] == tally_winner[valid_mask])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.4887 (var=0.0035)
**Predicted under pi_4:** 0.5097 (var=0.0037)
**Predicted under pi_5:** 0.5082 (var=0.0019)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Experiments 1 and 2 definitively falsify THEORY 2 (pi_5). The metric measures the average absolute deviation of subject-level choice proportions from 0.5. THEORY 2 predicted a value of ~0.25, consistent with its assumption of strong, fixed idiosyncratic side biases (where some subjects always pick A, others always pick B). The observed values (~0.04 to 0.05) tightly match the predictions of THEORY 1 (pi_4), which posits trial-by-trial random guessing (a binomial distribution with p=0.5 and n=96 trials yields an expected absolute deviation of ~0.04). Across all other experiments, both theories predict metrics around 0.5, but THEORY 1 accurately captures the within-subject variance. THEORY 2 is fundamentally degenerate because it assumes fixed individual-level biases that do not exist in the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory to replace the falsified idiosyncratic side bias theory. Since THEORY 1 (massive lapse rate / trial-by-trial guessing) fits the aggregate and variance data exceptionally well, the new theory should explore a different mechanism that produces near-random behavior. For instance, propose a 'Noisy Attribute' or 'Probabilistic Cue' theory where subjects attempt to use the provided feature validities (e.g., a noisy Take-The-Best or weighted additive model) but suffer from extremely high decision noise (low inverse temperature) or subjective cue validities that drift trial-to-trial. This would provide a structured, bounded-rational alternative to THEORY 1's pure lapse rate mechanism."
}
```

## Usage

```json
{
  "prompt_token_count": 6847,
  "candidates_token_count": 373,
  "total_token_count": 8342
}
```
