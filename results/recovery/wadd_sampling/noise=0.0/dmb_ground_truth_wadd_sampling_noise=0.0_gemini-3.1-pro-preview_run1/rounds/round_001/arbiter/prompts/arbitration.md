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
Weighted Additive (WADD) strategy: Decision-makers evaluate options by integrating all available information, computing a weighted sum of the features for each option where the weights are the cue validities. This compensatory approach balances the number of features an option possesses against the importance (validity) of those features.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design dissociates the Weighted Additive (WADD) strategy from Tallying by pitting a small number of high-validity features against a larger number of low-validity features. WADD computes a compensatory weighted sum, allowing strong cues to override a numerical majority of weaker cues. Tallying, by contrast, simply counts the number of winning features, ignoring validities entirely. The trial pairs include cases where WADD and Tallying make opposite predictions, cases where Tallying predicts a tie while WADD has a clear preference, and baseline cases where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Focuses strictly on the trial pairs where Weighted Additive and Tallying make diametrically opposite predictions. In these trials, one option possesses two high-validity features while the other possesses three low-validity features. WADD strongly prefers the option with the high-validity features, while Tallying prefers the option with more features. The metric calculates the proportion of choices aligned with WADD's prediction on these specific trials, yielding a value close to 1 for WADD and close to 0 for Tallying.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    wadd_aligned = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial type 1: A has the 2 high-validity features, B has the 3 low-validity features
        if a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            wadd_aligned.append(1.0 if resp == 0 else 0.0)
        # Trial type 2: Flipped
        elif a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            wadd_aligned.append(1.0 if resp == 1 else 0.0)
            
    if len(wadd_aligned) == 0:
        return 0.5
    return float(np.mean(wadd_aligned))

### RESULTS
- Predicted under pi_3 (simulated): 0.7700 (var=0.0188)
- Predicted under pi_2 (simulated): 0.1608 (var=0.0196)
- Observed on real data: 0.5083 (var=0.0801)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=7):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** This design dissociates Tallying from Weighted Additive (WADD) by contrasting the raw count of winning features against the sum of feature validities. By using five features with a steep drop-off in validity (two highly valid, three weakly valid), we can create scenarios where one option wins on a majority of features (favoring Tallying) while the other option wins on fewer but much more valid features (favoring WADD). We include trials where the models strongly disagree, trials where Tallying predicts a tie but WADD predicts a clear winner, and baseline trials where both models agree.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
The metric computes the proportion of choices that are consistent with the Tallying heuristic specifically on trials where Tallying and Weighted Additive (WADD) models make starkly opposing predictions. On these trials, Tallying counts the majority of weak features to pick one option, while WADD sums the fewer but highly valid features to pick the other. Tallying will yield a high proportion (near 1.0, modulo noise), while WADD will yield a low proportion (near 0.0).

Source:
def metric(data: pd.DataFrame) -> float:
    tallying_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_consistent += 1
            total += 1
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_consistent += 1
            total += 1
            
    if total == 0:
        return 0.5
    return tallying_consistent / total

### RESULTS
- Predicted under pi_3 (simulated): 0.2100 (var=0.0133)
- Predicted under pi_2 (simulated): 0.8769 (var=0.0076)
- Observed on real data: 0.3446 (var=0.0522)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def get_ttb_choice(a, b):
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return -1

    ttb_choices = [get_ttb_choice(a, b) for a, b in zip(data['option_a_ratings'], data['option_b_ratings'])]
    
    matches = (data['response'] == ttb_choices)
    return float(matches.mean())
```

**Observed (real) value:** 0.2758 (var=0.0200)
**Predicted under pi_3:** 0.1548 (var=0.0068)
**Predicted under pi_2:** 0.1369 (var=0.0088)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_ttb_match(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        for j in range(len(a)):
            if a[j] > b[j]:
                return resp == 0
            elif b[j] > a[j]:
                return resp == 1
        return False
        
    return float(data.apply(is_ttb_match, axis=1).mean())
```

**Observed (real) value:** 0.2825 (var=0.0090)
**Predicted under pi_3:** 0.2298 (var=0.0054)
**Predicted under pi_2:** 0.2248 (var=0.0047)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, neither the purely compensatory Weighted Additive (WADD, Theory 1) nor the purely non-compensatory Tallying (Theory 2) adequately captures the observed data. In Experiment 1, the observed proportion of WADD-aligned choices on conflict trials is ~0.51, exactly between WADD's prediction (0.77) and Tallying's prediction (0.16). In Experiment 2, Tallying-consistent choices are ~0.34, which is much closer to WADD's prediction (0.21) than Tallying's (0.88), but still significantly off. Experiments 3 and 4 also show that a pure Take-The-Best strategy is not being used, as TTB-consistency is very low (~0.28). The data suggests subjects are using a strategy that falls between the extreme sensitivity to exact validities (WADD) and complete insensitivity (Tallying).",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Tallying theory with a 'Rank-Weighted Additive' (or Validity-Ranked) theory. Instead of using the exact numerical validities (which allows top cues to dominate too much) or ignoring them completely (Tallying), this theory posits that decision-makers weight features by their ordinal rank of importance (e.g., weights of 5, 4, 3, 2, 1 for 5 features). This naturally compresses the validity scale, resulting in predictions that sit precisely between pure WADD and Tallying, matching the observed ~0.50 split in conflict trials."
}
```

## Usage

```json
{
  "prompt_token_count": 4842,
  "candidates_token_count": 375,
  "total_token_count": 6067
}
```
