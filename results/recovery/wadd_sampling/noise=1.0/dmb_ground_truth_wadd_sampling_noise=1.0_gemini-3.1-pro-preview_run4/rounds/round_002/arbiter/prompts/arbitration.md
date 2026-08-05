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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.51]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]

**Rationale:** This design systematically varies the tally difference between Option A and Option B across a wide range (from a 6-0 sweep to a 4-2 narrow win, and their inverses) to sharply distinguish the Advocated Theory (high noise/random guessing) from the Competing Theory (Tallying). Tallying, constrained by its parameter bounds (epsilon <= 0.5, beta >= 0.1), must predict a monotonically increasing preference for the option with more winning features, with a substantial choice probability for a 6-0 sweep. In contrast, the Advocated Theory predicts that due to massive lapse rates (epsilon >= 0.8), choice probabilities will remain virtually flat and near 50/50 across all trials, completely insensitive to the overwhelming feature evidence. A flat empirical choice curve would thus be uniquely predicted by the Advocated Theory.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design contains trials where one option consistently wins on a strict majority of features (e.g., 6-0, 5-1, or 4-2). The Competing Theory (Tallying) predicts that subjects will predominantly choose the option with more winning features, constrained by a relatively low lapse rate (epsilon <= 0.5), yielding an expected agreement rate of 0.75 or higher. In contrast, the Advocated Theory posits that subjects' choices are overwhelmingly dominated by noise or a massive lapse rate (epsilon >= 0.8), driving their choice probability near 50% regardless of the feature evidence. By calculating the proportion of trials on which the subject's choice aligns with the tallying majority, we can sharply discriminate the two theories: the metric will be near 0.5 for the Advocated Theory and >= 0.75 for the Competing Theory.

Source:
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


### RESULTS
- Predicted under pi_4 (simulated): 0.5262 (var=0.0019)
- Predicted under pi_2 (simulated): 0.8606 (var=0.0060)
- Observed on real data: 0.5021 (var=0.0041)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.85, 0.8, 0.75, 0.7, 0.6, 0.55]

**Trial pairs (n=7):**
  trial 1: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 3: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[0, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 1, 1]
  trial 6: A=[0, 0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 1, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 1]

**Rationale:** This design systematically varies the tally difference between Option A and Option B (from a 6-0 sweep down to a 3-3 tie, and the reverse) to quantitatively dissociate the Advocated Theory (Tallying) from the Competing Theory (Random Guessing). The Advocated Theory, characterized by low lapse rates and moderate-to-high beta, predicts a strong, monotonically increasing preference for the option with more winning features, reaching near-deterministic choices for large tally differences. In contrast, the Competing Theory, dominated by massive lapse rates (epsilon >= 0.8), predicts that choice probabilities will remain virtually flat and near 50/50 across all trials, entirely insensitive to overwhelming feature evidence. Observing a steep empirical choice curve would thus uniquely support the Advocated Theory.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the tally winner on trials where there is a strict tally difference (i.e., excluding ties). The Advocated Theory (Tallying) with moderate noise predicts a high match rate (significantly above 0.5), whereas the Competing Theory (Random Guessing) predicts a match rate very close to 0.5 due to its extremely high lapse rate. Because the number of non-tie trials is large, both theories will produce stable estimates per subject, leading to a highly significant t-test.

Source:
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

### RESULTS
- Predicted under pi_4 (simulated): 0.5097 (var=0.0037)
- Predicted under pi_2 (simulated): 0.8918 (var=0.0073)
- Observed on real data: 0.4887 (var=0.0035)

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
**Predicted under pi_2:** 0.1288 (var=0.0054)

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
**Predicted under pi_2:** 0.8623 (var=0.0092)

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
**Predicted under pi_2:** 0.5067 (var=0.0049)

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
**Predicted under pi_2:** 0.5271 (var=0.0046)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, the observed metric values consistently hover around 0.5. Theory 1 (pi_4), which posits massive response noise and random guessing, accurately predicts this near-chance performance across the board. Theory 2 (pi_2), which assumes subjects use a systematic Tallying heuristic with low noise, completely fails to capture the data, consistently predicting high alignment (e.g., ~0.86-0.89) or strong divergence when Tallying and the metric's focus diverge. It is clear that subjects are not systematically applying Tallying or Take-The-Best.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Theory 2 should be completely replaced. Since Theory 1 already captures the ~50% choice behavior via a massive lapse rate, the new Theory 2 should explore alternative mechanisms that also result in near-chance aggregate performance without relying on pure uniform guessing. For instance, propose a theory where subjects exhibit strong, idiosyncratic side biases (e.g., consistently picking Option A or Option B regardless of features), or a 'random single-feature' heuristic where subjects pick exactly one random feature per trial to base their decision on. Both would yield ~0.5 on systematic multi-cue metrics but imply a different underlying cognitive process than simple lapse-driven guessing."
}
```

## Usage

```json
{
  "prompt_token_count": 6420,
  "candidates_token_count": 300,
  "total_token_count": 7751
}
```
