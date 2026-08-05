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
Take-K Heuristic: Decision-makers sort cues by validity but only tally the wins on the top K most valid cues, ignoring the rest. This blends the frugality of Take-The-Best with the compensatory accumulation of Tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    n_features = len(val)
    
    # K is the number of top cues to consider, determined by k_prop
    k_prop = float(parameters["k_prop"])
    k = max(1, int(np.round(k_prop * n_features)))
    
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable")
    top_cues = cue_order[:k]
    
    a_top = a[top_cues]
    b_top = b[top_cues]
    
    # Tally strict wins on the top K cues
    a_wins = float(np.sum(a_top > b_top))
    b_wins = float(np.sum(b_top > a_top))
    
    scores = np.array([a_wins, b_wins])
    
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


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.98, 0.92, 0.86, 0.8, 0.74, 0.51]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 2: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 4: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 6: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 0, 0]

**Rationale:** To dissociate the Take-K heuristic from Tallying, we exploit their differential treatment of the lowest-validity cues. Take-K sorts cues by validity and ignores the bottom fraction (depending on its k_prop parameter), whereas Tallying counts wins across all cues regardless of validity. We use a 6-feature design where the 6th cue has a substantially lower validity than the rest. The trials are constructed so that the top 5 cues result in a tie (which Take-K will guess on) but the 6th cue breaks the tie (which Tallying will follow), or conversely, the top 5 cues favor one option but the 6th cue balances the overall tally (leading Tallying to guess, while Take-K makes a deterministic choice).

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the top 5 cues strictly favor one option, but the 6th cue favors the other option, resulting in an overall tie across all 6 cues (Trials 2, 3, and 5). Tallying considers all 6 cues, sees a tie, and is forced to guess (yielding ~0.50). Take-K, depending on its k_prop parameter, will either ignore the 6th cue and deterministically choose the top-5 winner, or include it and guess. This produces a reliable mean difference between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    match_count = 0
    total_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_top5 = a[:5]
        b_top5 = b[:5]
        
        a_wins = np.sum(a_top5 > b_top5)
        b_wins = np.sum(b_top5 > a_top5)
        
        if a_wins > b_wins:
            if row['response'] == 0:
                match_count += 1
            total_count += 1
        elif b_wins > a_wins:
            if row['response'] == 1:
                match_count += 1
            total_count += 1
            
    if total_count == 0:
        return 0.5
    return float(match_count / total_count)

### RESULTS
- Predicted under pi_4 (simulated): 0.7113 (var=0.0562)
- Predicted under pi_2 (simulated): 0.5008 (var=0.0051)
- Observed on real data: 0.1717 (var=0.0110)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=7):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.55]

**Trial pairs (n=7):**
  trial 1: A=[1, 1, 1, 0, 0, 0, 1]  B=[0, 0, 0, 1, 1, 1, 0]
  trial 2: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 1, 0, 1]  B=[0, 0, 1, 1, 0, 1, 0]
  trial 4: A=[1, 1, 0, 0, 1, 0, 0]  B=[0, 0, 1, 1, 0, 1, 1]
  trial 5: A=[1, 1, 1, 1, 0, 0, 1]  B=[0, 0, 0, 0, 1, 1, 0]
  trial 6: A=[1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0]

**Rationale:** To dissociate Tallying (the advocated theory) from the Take-K heuristic, we exploit both the structural difference in cue usage and the bounds on their noise parameters. Take-K sorts cues by validity and may discard the lowest-validity cues (depending on its k_prop parameter), whereas Tallying counts wins across all cues equally. Furthermore, Take-K is constrained to have high determinism (beta >= 2.0, epsilon <= 0.1), while Tallying allows for much softer, noisier choices (beta down to 0.1, epsilon up to 0.5). We use a 7-feature design where the 7th cue has the lowest validity. In key trials, the top 6 cues result in a perfect tie (3 wins each), but the 7th cue breaks the tie. Take-K (if k_prop < 1.0) will ignore the 7th cue and predict a 50/50 guess, while Tallying will use the 7th cue to predict a preference. Even if Take-K sets k_prop = 1.0 to mimic Tallying, its high-beta constraint forces it to predict very sharp probabilities (>= 88% for a 1-win difference), whereas Tallying can correctly capture a much softer empirical preference (e.g., 60%) driven by the small 1-cue tally difference.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



### METRIC
Rationale:
This metric calculates the choice accuracy (relative to the full unweighted tally) specifically on trials where the difference in tally scores is large (>= 2). Take-K is constrained to have high determinism (beta >= 2.0) and low lapse rates (epsilon <= 0.1), which forces it to make highly consistent choices on these 'easy' trials regardless of whether k_prop dictates K=6 or K=7. Tallying, however, allows for a much broader range of noise parameters (beta down to 0.1, epsilon up to 0.5), meaning that on average, data generated under Tallying will show a significantly lower rate of choosing the tally-maximizing option even when the tally difference is large. This creates a strong, stable contrast between the theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    correct_count = 0
    total_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials with a strong tally difference (>= 2)
        if abs(a_wins - b_wins) >= 2:
            total_count += 1
            if a_wins > b_wins and row['response'] == 0:
                correct_count += 1
            elif b_wins > a_wins and row['response'] == 1:
                correct_count += 1
                
    if total_count == 0:
        return 0.0
    return float(correct_count / total_count)

### RESULTS
- Predicted under pi_4 (simulated): 0.9754 (var=0.0009)
- Predicted under pi_2 (simulated): 0.8600 (var=0.0131)
- Observed on real data: 0.8554 (var=0.0133)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb = 0
                break
            elif b[i] > a[i]:
                ttb = 1
                break
        
        if ttb is not None:
            matches.append(1.0 if row['response'] == ttb else 0.0)
            
    if not matches:
        return 0.5
    return float(np.mean(matches))
```

**Observed (real) value:** 0.1454 (var=0.0092)
**Predicted under pi_4:** 0.3756 (var=0.0272)
**Predicted under pi_2:** 0.3196 (var=0.0022)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a = np.stack(data['option_a_ratings'].values)
    b = np.stack(data['option_b_ratings'].values)
    a_wins = np.sum(a > b, axis=1)
    b_wins = np.sum(b > a, axis=1)
    
    valid = a_wins != b_wins
    if not np.any(valid):
        return 0.5
    
    pred = (b_wins[valid] > a_wins[valid]).astype(int)
    resp = data['response'].values[valid]
    
    return float(np.mean(pred == resp))
```

**Observed (real) value:** 0.7971 (var=0.0103)
**Predicted under pi_4:** 0.8731 (var=0.0191)
**Predicted under pi_2:** 0.8707 (var=0.0085)

### Experiment 5
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    val = np.array([0.95, 0.85, 0.65, 0.55, 0.5])
    
    wadd_consistent_choices = []
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        wadd_a = np.sum(a * val)
        wadd_b = np.sum(b * val)
        
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        
        # Check if WADD and Tallying make strictly opposite predictions
        if (wadd_a > wadd_b and tally_a < tally_b) or (wadd_a < wadd_b and tally_a > tally_b):
            wadd_pref = 0 if wadd_a > wadd_b else 1
            if row['response'] == wadd_pref:
                wadd_consistent_choices.append(1)
            else:
                wadd_consistent_choices.append(0)
                
    if len(wadd_consistent_choices) == 0:
        return 0.5
    return float(np.mean(wadd_consistent_choices))
```

**Observed (real) value:** 0.1733 (var=0.0221)
**Predicted under pi_4:** 0.1600 (var=0.0444)
**Predicted under pi_2:** 0.1411 (var=0.0136)

### Experiment 6
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where Tallying has a strict preference
    # (i.e., one option has more positive ratings than the other)
    mask = sum_a != sum_b
    if not mask.any():
        return 0.5
        
    # Tallying predicts choosing the option with the higher sum.
    # Response is 0 for A, 1 for B.
    # If sum_a < sum_b, Tallying prefers B (1).
    # If sum_a > sum_b, Tallying prefers A (0).
    tallying_choice = (sum_a < sum_b).astype(int)
    
    # Calculate the proportion of responses matching the Tallying prediction
    matches = (data.loc[mask, 'response'] == tallying_choice.loc[mask])
    
    return float(matches.mean())
```

**Observed (real) value:** 0.8125 (var=0.0197)
**Predicted under pi_4:** 0.8444 (var=0.0501)
**Predicted under pi_2:** 0.8450 (var=0.0103)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, subjects consistently follow the Tallying heuristic when one option has strictly more feature wins than the other. This is evident in Experiments 3, 4, 5, and 6, where subjects overwhelmingly choose the Tallying winner even when it conflicts with Take-The-Best (Exp 3) or a Weighted Additive model (Exp 5). However, Experiment 1 reveals a critical blind spot in the pure Tallying theory (pi_2): when the tally results in a tie, subjects do not simply guess (as pi_2 predicts with a 0.50 metric). Instead, they systematically break the tie in favor of the option that has the higher-validity cues (yielding an observed metric of 0.17, far from 0.50). Neither pure Tallying nor Take-K (pi_4) captures this two-stage lexicographic process. Take-K (pi_4) fails because it truncates cues rather than using them as a fallback.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Tally-then-TTB' (or 'Tally with Validity Tie-Breaker') theory. The theory should state that decision-makers first compare options by tallying the total number of winning features for each. If one option has strictly more wins, it is chosen. If the tally results in a tie, the decision-maker falls back to Take-The-Best, breaking the tie by choosing the option that wins on the single most valid cue. This preserves the strong Tallying performance on unequal-tally trials while accurately capturing the deterministic tie-breaking behavior observed in Experiment 1."
}
```

## Usage

```json
{
  "prompt_token_count": 6773,
  "candidates_token_count": 378,
  "total_token_count": 10189
}
```
