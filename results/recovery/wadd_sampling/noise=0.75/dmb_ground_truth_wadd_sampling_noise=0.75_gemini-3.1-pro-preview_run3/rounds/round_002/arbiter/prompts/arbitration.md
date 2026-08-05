# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_2_1") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2_1" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People make decisions by computing a weighted sum of the features for each option, where the weights correspond to a non-linear scaling of the subjective validities of the cues. This Weighted Additive (WADD) strategy allows for compensatory decision-making, where multiple weak cues can override a single strong cue, but the non-linear scaling (gamma parameter) flexibly tunes how strongly higher-validity cues dominate over weaker ones. A softmax choice rule on these weighted sums, along with a lapse rate, introduces response noise and accounts for the intermediate choice patterns observed in human data.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    # Calculate weighted sum for each option
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    scores = np.array([score_a, score_b])
    
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
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_2_1
People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
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
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design provides a granular dissociation of the WADD (Weighted Additive) and Tallying models by varying the trade-off between the number of winning features and their validities. Tallying purely counts the number of winning features, ignoring validities, leading to ties when counts are equal and strict preferences when one option wins on more features. In contrast, WADD scales features by a non-linear function of their validities. By distributing five features with linearly spaced validities, we create trials where Tallying prefers an option with a higher count of low-validity features, while WADD prefers the option with fewer, but higher-validity features. We also include trials where Tallying predicts a tie but WADD predicts a strong preference, and trials where both models agree on the direction but differ in the magnitude of preference, allowing for precise parameter estimation of WADD's gamma scaling and choice noise.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
Tallying decides based purely on the unweighted count of features. In Trial 1 and Trial 2, Option B has exactly 2 more winning features than Option A, so Tallying predicts the exact same probability of choosing B for both trials. Similarly, in Trial 7 and Trial 8, Option B has exactly 1 more winning feature than A, leading to identical choice probabilities under Tallying. Therefore, the metric (P(B|T1) - P(B|T2)) + (P(B|T7) - P(B|T8)) has an expected value of exactly 0 for Tallying. Conversely, WADD weights features by their subjective validities. Because Option B's features in Trial 1 have higher validities than in Trial 2, and similarly for Trial 7 versus Trial 8, WADD strongly predicts a higher probability of choosing B in the first trial of each pair. This produces a strictly positive expected value for WADD, effectively isolating the difference in how the two models treat feature validities while controlling for raw feature counts.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Project lists to strings for hashable matching
    a_str = data['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    # Trial 1: A=[1,0,0,0,0], B=[0,1,1,1,0]. Tallying tally diff = 3 - 1 = 2.
    t1_mask = (a_str == "10000") & (b_str == "01110")
    # Trial 2: A=[1,0,0,0,0], B=[0,0,1,1,1]. Tallying tally diff = 3 - 1 = 2.
    t2_mask = (a_str == "10000") & (b_str == "00111")
    
    # Trial 7: A=[1,0,1,0,0], B=[0,1,0,1,1]. Tallying tally diff = 3 - 2 = 1.
    t7_mask = (a_str == "10100") & (b_str == "01011")
    # Trial 8: A=[1,1,0,0,0], B=[0,0,1,1,1]. Tallying tally diff = 3 - 2 = 1.
    t8_mask = (a_str == "11000") & (b_str == "00111")
    
    p_b_t1 = data.loc[t1_mask, 'response'].mean()
    p_b_t2 = data.loc[t2_mask, 'response'].mean()
    p_b_t7 = data.loc[t7_mask, 'response'].mean()
    p_b_t8 = data.loc[t8_mask, 'response'].mean()
    
    # Handle missing trial types gracefully
    p_b_t1 = 0.5 if pd.isna(p_b_t1) else p_b_t1
    p_b_t2 = 0.5 if pd.isna(p_b_t2) else p_b_t2
    p_b_t7 = 0.5 if pd.isna(p_b_t7) else p_b_t7
    p_b_t8 = 0.5 if pd.isna(p_b_t8) else p_b_t8
    
    # Tallying predicts exactly 0 difference for both pairs. 
    # WADD predicts positive differences because B has higher validity features in T1 vs T2, and T7 vs T8.
    return float((p_b_t1 - p_b_t2) + (p_b_t7 - p_b_t8))

### RESULTS
- Predicted under pi_3 (simulated): 0.2900 (var=0.0917)
- Predicted under pi_2_1 (simulated): -0.0417 (var=0.0484)
- Observed on real data: -0.0700 (var=0.1048)

## EXPERIMENT 2 (proposed by pi_2_1)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.8, 0.65, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 3: A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]
  trial 8: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate Tallying from WADD, we use a 6-feature design with a pronounced gap between high-validity and low-validity cues. Tallying relies exclusively on the count of winning features, predicting a tie when the counts are equal and a strict preference when one option has more wins, completely ignoring the validities. WADD, however, computes a weighted sum scaled by validities. By including trials where one option wins on few high-validity cues while the other wins on many low-validity cues, Tallying and WADD will predict opposing choices. Additionally, trials where both options win on an equal number of features but differ drastically in their validities will yield a 50-50 guess under Tallying but a strong deterministic preference under WADD.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric contrasts the probability of choosing Option A on Trial 1 vs Trial 4. In both trials, Option A and Option B win on an equal number of features (2 vs 2 in Trial 1, 3 vs 3 in Trial 4), so the Tallying heuristic views both as exact ties and predicts a 50% chance of choosing Option A in both cases, leading to a difference of 0. However, in Trial 1, Option A wins on the highest validity cues, whereas in Trial 4, Option B wins on the highest validity cues. WADD is highly sensitive to these validities and will strongly prefer Option A in Trial 1 and Option B in Trial 4, resulting in a large positive difference. This provides a clear, quantitative dissociation between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Identify Trial 1: A=[1, 1, 0, 0, 0, 0], B=[0, 0, 0, 1, 1, 0]
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0, 0)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 0))
    
    # Identify Trial 4: A=[0, 0, 0, 1, 1, 1], B=[1, 1, 1, 0, 0, 0]
    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1, 1)) & \
              data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 1, 0, 0, 0))
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    # Handle edge cases where a subject might have missed these trials (should not happen in full design)
    if pd.isna(p_a_t1): p_a_t1 = 0.5
    if pd.isna(p_a_t4): p_a_t4 = 0.5
    
    return float(p_a_t1 - p_a_t4)


### RESULTS
- Predicted under pi_3 (simulated): 0.6567 (var=0.0541)
- Predicted under pi_2_1 (simulated): -0.0100 (var=0.0405)
- Observed on real data: 0.0800 (var=0.0525)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
import pandas as pd
import numpy as np

def metric(data: pd.DataFrame) -> float:
    def ttb_choice(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        for i in range(len(a)):
            if a[i] > b[i]:
                return 0
            elif b[i] > a[i]:
                return 1
        return 0.5
    
    ttb_preds = data.apply(ttb_choice, axis=1)
    return float(np.mean(data['response'] == ttb_preds))
```

**Observed (real) value:** 0.4425 (var=0.0035)
**Predicted under pi_3:** 0.5642 (var=0.0484)
**Predicted under pi_2_1:** 0.4529 (var=0.0047)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_aligned_choices = 0
    disagreement_trials = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        if tally_a > tally_b:
            tally_pred = 0
        elif tally_b > tally_a:
            tally_pred = 1
        else:
            tally_pred = None
            
        ttb_pred = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_pred = 0
                break
            elif b[i] > a[i]:
                ttb_pred = 1
                break
                
        if tally_pred is not None and ttb_pred is not None and tally_pred != ttb_pred:
            disagreement_trials += 1
            if row['response'] == ttb_pred:
                ttb_aligned_choices += 1
                
    if disagreement_trials == 0:
        return 0.5
        
    return float(ttb_aligned_choices / disagreement_trials)
```

**Observed (real) value:** 0.4578 (var=0.0041)
**Predicted under pi_3:** 0.4869 (var=0.0441)
**Predicted under pi_2_1:** 0.4439 (var=0.0058)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    opt1 = (1, 1, 0, 0, 0)
    opt2 = (0, 0, 1, 1, 0)
    
    chose_opt1 = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == opt1 and b == opt2:
            chose_opt1.append(1 if row['response'] == 0 else 0)
        elif a == opt2 and b == opt1:
            chose_opt1.append(1 if row['response'] == 1 else 0)
            
    if not chose_opt1:
        return 0.5
    return sum(chose_opt1) / len(chose_opt1)
```

**Observed (real) value:** 0.5100 (var=0.0385)
**Predicted under pi_3:** 0.8183 (var=0.0255)
**Predicted under pi_2_1:** 0.4667 (var=0.0156)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = data['A_str'] == '11000'
    t4_mask = data['A_str'] == '00011'
    
    # response == 0 means subject chose A
    p_A_t1 = 1.0 - data.loc[t1_mask, 'response'].mean()
    p_A_t4 = 1.0 - data.loc[t4_mask, 'response'].mean()
    
    if pd.isna(p_A_t1): p_A_t1 = 0.5
    if pd.isna(p_A_t4): p_A_t4 = 0.5
    
    return float(p_A_t1 - p_A_t4)
```

**Observed (real) value:** 0.0025 (var=0.0611)
**Predicted under pi_3:** 0.6450 (var=0.0596)
**Predicted under pi_2_1:** 0.0075 (var=0.0281)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, Theory 2 (Tallying) dramatically outperforms Theory 1 (WADD). The observed metrics consistently hover around the values predicted by Tallying (e.g., near 0 for differences in choices between tied tally counts, regardless of feature validities), whereas WADD consistently predicts large differences due to its reliance on validity weighting. Subjects appear to be ignoring the cue validities and simply counting the number of winning features. Theory 1 is degenerate because its core assumption\u2014that humans compute a compensatory weighted sum of validities\u2014is strongly contradicted by the data.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace the WADD theory with the 'Take-The-Best' (TTB) heuristic. TTB offers a classic, non-compensatory alternative to Tallying. Under TTB, the decision-maker considers cues one at a time in descending order of validity, stopping at the first cue that discriminates between the two options and choosing the option favored by that cue. This will provide a strong non-compensatory baseline that respects cue ordering, contrasting perfectly with Tallying (which is compensatory but ignores cue ordering)."
}
```

## Usage

```json
{
  "prompt_token_count": 7415,
  "candidates_token_count": 265,
  "total_token_count": 8611
}
```
