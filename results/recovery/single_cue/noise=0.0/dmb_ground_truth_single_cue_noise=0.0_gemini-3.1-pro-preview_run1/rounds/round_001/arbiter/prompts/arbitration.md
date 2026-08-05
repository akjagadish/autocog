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
People make decisions by computing a weighted sum of the features for each option, using the provided cue validities as weights (Weighted Additive model). Unlike non-compensatory heuristics like Take The Best or unweighted ones like Tallying, WADD integrates all available information and scales each feature by its explicit validity. The option with the higher total weighted score is preferred, subject to decision noise (softmax) and occasional lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute weighted sum of features for each option
    a_score = np.dot(a, val)
    b_score = np.dot(b, val)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=float)
    probs /= np.sum(probs)
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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 8: A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) model from Tallying, we exploit their divergent treatment of cue validities. Tallying treats all features equally, merely counting the number of features on which one option strictly dominates the other. WADD, by contrast, weights each feature's contribution by its explicitly provided validity. We employ 5 binary features with a steep drop-off in validities (e.g., two highly valid cues and three less valid cues). The critical trials pair an option that wins on the two highly valid cues against an option that wins on the three less valid cues. Tallying predicts a preference for the latter (3 wins vs 2 wins), whereas WADD predicts a preference for the former due to the larger summed validity. We also include trials where the options tie in the number of winning features (leading Tallying to guess uniformly) but differ in their weighted sums, allowing WADD to make a deterministic prediction.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The metric computes the proportion of choices that align with the Weighted Additive (WADD) model on the subset of critical trials where WADD and Tallying make diametrically opposed predictions. Specifically, in trials where one option has 2 highly valid features and the other has 3 less valid features, WADD predicts a preference for the 2-feature option (due to a higher weighted sum), while Tallying predicts a preference for the 3-feature option (due to a higher unweighted count). This metric will yield values close to 1.0 for WADD and close to 0.0 for Tallying, providing a massive and stable contrast.

Source:
def metric(data: pd.DataFrame) -> float:
    wadd_consistent = 0
    total = 0
    for _, row in data.iterrows():
        a_sum = sum(row['option_a_ratings'])
        b_sum = sum(row['option_b_ratings'])
        if a_sum == 2 and b_sum == 3:
            if row['response'] == 0:
                wadd_consistent += 1
            total += 1
        elif a_sum == 3 and b_sum == 2:
            if row['response'] == 1:
                wadd_consistent += 1
            total += 1
    if total == 0:
        return 0.5
    return wadd_consistent / total

### RESULTS
- Predicted under pi_3 (simulated): 0.7250 (var=0.0197)
- Predicted under pi_2 (simulated): 0.1208 (var=0.0082)
- Observed on real data: 0.1400 (var=0.0117)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.95, 0.55, 0.55, 0.55]

**Trial pairs (n=7):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate Tallying from WADD, we use 5 binary features where two cues have very high validity (0.95) and three have low validity (0.55). Tallying ignores validities and simply counts the number of winning features, whereas WADD integrates the validities into a weighted sum. The critical dissociation trials pit an option that wins on the three low-validity features against an option that wins on the two high-validity features. Tallying predicts a preference for the former (3 wins vs 2 wins), whereas WADD predicts a strong preference for the latter (weighted sum 1.90 vs 1.65). We also include trials where Tallying results in a tie (equal number of winning features) but WADD has a strong preference, as well as baseline trials where both models agree.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that are consistent with the Tallying heuristic strictly on the critical dissociation trials (Trials 1 and 2). In these trials, one option has 3 low-validity features (winning the tally 3-to-2) while the other has 2 high-validity features (winning the weighted sum 1.90 to 1.65). Tallying strongly predicts choosing the option with 3 features, whereas WADD strongly predicts choosing the option with 2 high-validity features. The metric will evaluate to a high value (near 1) under Tallying and a low value (near 0) under WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A_t1 = [0, 0, 1, 1, 1]
    B_t1 = [1, 1, 0, 0, 0]
    A_t2 = [1, 1, 0, 0, 0]
    B_t2 = [0, 0, 1, 1, 1]
    
    consistencies = []
    
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        r = row['response']
        
        # Trial 1: Tallying prefers A (3 wins vs 2), WADD prefers B (1.90 vs 1.65)
        if a == A_t1 and b == B_t1:
            consistencies.append(1 if r == 0 else 0)
        # Trial 2: Tallying prefers B (3 wins vs 2), WADD prefers A (1.90 vs 1.65)
        elif a == A_t2 and b == B_t2:
            consistencies.append(1 if r == 1 else 0)
            
    if not consistencies:
        return 0.5
    return float(np.mean(consistencies))

### RESULTS
- Predicted under pi_3 (simulated): 0.2362 (var=0.0189)
- Predicted under pi_2 (simulated): 0.8846 (var=0.0107)
- Observed on real data: 0.8354 (var=0.0165)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    match_ttb = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # TTB prediction: first discriminating cue
        ttb_winner = None
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_winner = 0
                break
            elif b[j] > a[j]:
                ttb_winner = 1
                break
                
        # Tallying prediction: majority of discriminating cues
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        if a_wins > b_wins:
            tally_winner = 0
        elif b_wins > a_wins:
            tally_winner = 1
        else:
            tally_winner = None
            
        # Only consider trials where the two heuristics deterministically disagree
        if ttb_winner is not None and tally_winner is not None and ttb_winner != tally_winner:
            if row['response'] == ttb_winner:
                match_ttb += 1
            total += 1
            
    if total == 0:
        return 0.5
    return match_ttb / total
```

**Observed (real) value:** 0.2775 (var=0.0072)
**Predicted under pi_3:** 0.1437 (var=0.0120)
**Predicted under pi_2:** 0.1317 (var=0.0087)

### Experiment 4
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  A=[0, 0, 1, 0]  B=[0, 1, 0, 0]
  A=[0, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    tally_match = 0
    total = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins != b_wins:
            tally_pref = 0 if a_wins > b_wins else 1
            if row['response'] == tally_pref:
                tally_match += 1
            total += 1
            
    return float(tally_match / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.8178 (var=0.0246)
**Predicted under pi_3:** 0.8550 (var=0.0087)
**Predicted under pi_2:** 0.8661 (var=0.0073)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across Experiments 1 and 2, the observed data strongly contradict the Weighted Additive (WADD) model (THEORY 1) and closely align with the Tallying heuristic (THEORY 2). When forced to choose between an option with fewer high-validity cues and one with more low-validity cues, participants overwhelmingly chose the option with more winning cues, ignoring the explicit validities. Experiment 4 also shows a high match with Tallying. Experiment 3 tests TTB vs Tallying, showing that choices align with TTB only ~28% of the time, meaning they align with Tallying ~72% of the time. While Tallying is clearly the dominant strategy, WADD is completely degenerate here. We should replace WADD with another classic decision-making heuristic to serve as a stronger structural contrast.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace the WADD theory with the 'Take The Best' (TTB) heuristic. TTB is a classic non-compensatory model that searches through cues in descending order of their validity. It stops at the very first cue that discriminates between the two options (where one option has a 1 and the other a 0) and chooses the option favored by that cue, ignoring all remaining cues. This will provide a sharp contrast to Tallying (which ignores validities but integrates across all cues) and allow us to model the minority of choices in Experiment 3 that deviated from Tallying."
}
```

## Usage

```json
{
  "prompt_token_count": 5407,
  "candidates_token_count": 337,
  "total_token_count": 6565
}
```
