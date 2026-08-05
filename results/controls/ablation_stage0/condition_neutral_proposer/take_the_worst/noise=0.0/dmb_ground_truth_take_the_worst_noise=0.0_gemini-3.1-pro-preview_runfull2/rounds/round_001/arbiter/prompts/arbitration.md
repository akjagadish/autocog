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
People make decisions between options by computing a fully compensatory weighted sum of their features, where each feature is weighted by its subjective or objective validity. This Weighted Additive (WADD) strategy contrasts with non-compensatory rules like Take The Best by allowing multiple weak cues to overcome a single strong cue, and contrasts with Tallying by taking cue validities into account. The resulting scores are transformed into choice probabilities via a softmax function with a lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match number of features.")
        
    a, b = stim[0], stim[1]
    
    # WADD computes the sum of feature values weighted by validities
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
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

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 9: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 10: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To maximally dissociate WADD from Tallying, we use 5 features with a steep drop-off in validities. WADD computes a weighted sum, meaning it can strongly prefer an option endorsed by a small number of high-validity features. Tallying discards validities and simply counts the number of winning features. By pitting 2 high-validity features against 3 low-validity features, we create strong dissociations where WADD prefers the option with 2 features while Tallying prefers the option with 3. We also include pairs with an equal number of winning features (which Tallying considers a tie) but where WADD has a clear preference due to the different validities, further pulling apart their predictions.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
By analyzing trials where WADD and Tallying make strictly opposing predictions (e.g., when 2 high-validity features outscore 3 low-validity features under WADD, but lose under Tallying's count), we can compute the proportion of choices that align with WADD. WADD will reliably score > 0.5 (closer to 1.0) on this metric, while Tallying will reliably score < 0.5 (closer to 0.0), producing a massive and stable gap between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    v = np.array([0.95, 0.9, 0.6, 0.55, 0.5])
    
    wadd_aligned = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        wadd_A = np.sum(a * v)
        wadd_B = np.sum(b * v)
        
        tally_A = np.sum(a > b)
        tally_B = np.sum(b > a)
        
        # Identify trials where WADD and Tallying make opposite predictions
        if wadd_A > wadd_B and tally_A < tally_B:
            wadd_aligned.append(1 if row['response'] == 0 else 0)
        elif wadd_A < wadd_B and tally_A > tally_B:
            wadd_aligned.append(1 if row['response'] == 1 else 0)
            
    if len(wadd_aligned) == 0:
        return 0.5
    return float(np.mean(wadd_aligned))

### RESULTS
- Predicted under pi_3 (simulated): 0.7667 (var=0.0244)
- Predicted under pi_2 (simulated): 0.1156 (var=0.0152)
- Observed on real data: 0.2067 (var=0.0216)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]

**Rationale:** To maximally dissociate Tallying from WADD, we use 5 features with a steep drop-off in validities. WADD computes a weighted sum and can strongly prefer an option endorsed by a small number of high-validity features. Tallying discards validities entirely, simply counting the number of winning features for each option. By pitting 2 high-validity features against 3 low-validity features, we create strong dissociations where WADD prefers the option with 2 features while Tallying prefers the option with 3. We also include pairs with an equal number of winning features (which Tallying considers a tie) but where WADD has a clear preference due to the different validities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric targets the critical dissociation trials (Trials 1 and 3) where Tallying and WADD have completely opposite preferences. In these trials, one option is endorsed by 2 high-validity features (summing to 1.8 for WADD) and the other by 3 lower-validity features (summing to 1.65). Tallying simply counts features and strictly prefers the option with 3 features. WADD computes the weighted sum and strictly prefers the option with 2 features. By calculating the proportion of choices for the option with 3 features on these specific trials, Tallying will predict a value > 0.5 (approaching 1.0), while WADD will predict a value < 0.5 (approaching 0.0).

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    # Isolate the critical trials where one option has 2 positive features and the other has 3
    mask = ((a_sums == 2) & (b_sums == 3)) | ((a_sums == 3) & (b_sums == 2))
    if not mask.any():
        return 0.5
    subset = data[mask]
    
    # Tallying prefers the option with 3 features; WADD prefers the option with 2 features 
    # (because the 2 features have higher validities: 0.95 + 0.85 = 1.8 vs 0.6 + 0.55 + 0.5 = 1.65)
    a_is_3 = subset['option_a_ratings'].apply(sum) == 3
    
    # response == 0 means choice A, response == 1 means choice B
    # We check if the subject chose the option with 3 features
    chose_tallying = (a_is_3.astype(int) == (1 - subset['response']))
    
    return float(chose_tallying.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.2883 (var=0.0203)
- Predicted under pi_2 (simulated): 0.8558 (var=0.0184)
- Observed on real data: 0.8433 (var=0.0173)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    ttb_preds = np.zeros(len(data))
    for i in range(len(data)):
        a = a_ratings[i]
        b = b_ratings[i]
        pred = 0.5
        for j in range(4):
            if a[j] > b[j]:
                pred = 0
                break
            elif b[j] > a[j]:
                pred = 1
                break
        ttb_preds[i] = pred
        
    matches = (responses == ttb_preds).mean()
    return float(matches)
```

**Observed (real) value:** 0.2213 (var=0.0039)
**Predicted under pi_3:** 0.3252 (var=0.0042)
**Predicted under pi_2:** 0.2956 (var=0.0061)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = 0
    count = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        a_arr = np.array(a)
        b_arr = np.array(b)
        
        a_wins = np.sum(a_arr > b_arr)
        b_wins = np.sum(b_arr > a_arr)
        
        if a_wins > b_wins:
            tally_pref = 0
        elif b_wins > a_wins:
            tally_pref = 1
        else:
            continue
            
        if resp == tally_pref:
            matches += 1
        count += 1
        
    return float(matches / count) if count > 0 else 0.5
```

**Observed (real) value:** 0.7294 (var=0.0080)
**Predicted under pi_3:** 0.8317 (var=0.0101)
**Predicted under pi_2:** 0.8592 (var=0.0113)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, THEORY 2 (Tallying) provides a vastly superior fit to the observed data compared to THEORY 1 (WADD). In Experiment 1, the observed metric measuring WADD-alignment was 0.2067, which strongly rejected WADD's prediction (0.7667) in favor of Tallying's prediction (0.1156). In Experiment 2, the observed metric measuring Tallying-alignment was 0.8433, almost perfectly matching Tallying's prediction (0.8558) and completely contradicting WADD (0.2883). Experiment 4 also showed high alignment with Tallying (0.7294). WADD's assumption of fully compensatory weighting using validities clearly fails here. However, Tallying was slightly overconfident in Experiment 4 and missed the exact TTB metric in Experiment 3. Because THEORY 2 is highly successful, it should be kept. THEORY 1 is degenerate and should be replaced with a new theory that can compete with Tallying but perhaps account for some bounded use of validities.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory that bridges the gap between purely unweighted Tallying and fully compensatory WADD. Since Tallying (ignoring validities) works well but isn't perfectly capturing the noise/variance, propose a 'Top-N Tallying' or 'Truncated Tallying' theory. In this theory, subjects only tally the features among the top k most valid cues (ignoring the lowest validity cues completely to save cognitive effort) and guess if they tie. This differs from Tallying by selectively ignoring low-validity features, and differs from WADD by still ignoring the relative cardinal weights among the considered features."
}
```

## Usage

```json
{
  "prompt_token_count": 5460,
  "candidates_token_count": 413,
  "total_token_count": 7100
}
```
