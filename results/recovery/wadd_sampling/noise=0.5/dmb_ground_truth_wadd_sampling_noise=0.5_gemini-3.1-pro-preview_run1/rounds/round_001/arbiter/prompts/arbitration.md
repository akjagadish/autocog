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
People make choices by computing a weighted sum of all available features for each option, but the weights they use are a non-linear transformation of the objective cue validities. By exponentiating the validities with a free parameter gamma, the strategy can smoothly interpolate between equal weighting (Tallying, gamma=0) and a strong reliance on the most valid cues (approximating Take The Best, gamma > 1). The choice is then made probabilistically based on the difference between the options' weighted sums, with a bounded inverse temperature beta to allow for more stochastic choices.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities
    subjective_weights = val ** gamma
    
    # Calculate weighted sums for both options
    a, b = stim[0], stim[1]
    score_a = np.sum(a * subjective_weights)
    score_b = np.sum(b * subjective_weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate response noise (lapse rate)
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure valid probabilities
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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Rationale:** To dissociate WADD from Tallying, we use a 5-feature design where features have a wide spread of validities. Tallying simply counts the number of features favoring each option, ignoring validities. WADD, however, weights features by a non-linear transformation of their validities. The trials are constructed so that one option has a numerical majority of winning features (favored by Tallying), while the other option wins on fewer but highly valid features (favored by WADD, depending on the gamma parameter). We also include trials where Tallying predicts a tie (equal number of winning features) but WADD strongly prefers the option with higher-validity features.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In Trial 6, the options are [1, 1, 0, 0, 0] and [0, 0, 1, 1, 0]. Tallying counts exactly 2 winning features for each option, predicting a tie (exactly 50% choice probability) regardless of its beta or epsilon parameters. WADD, however, assigns a higher subjective weight to the first two features (which have higher objective validities) for any gamma > 0, leading to a systematic preference for [1, 1, 0, 0, 0]. This metric computes the proportion of times [1, 1, 0, 0, 0] is chosen in this trial type. Tallying will yield a mean of 0.5 with minimal between-subject variance (only binomial noise), while WADD will yield a mean significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    target = (1, 1, 0, 0, 0)
    other = (0, 0, 1, 1, 0)
    a_keys = data['option_a_ratings'].apply(tuple)
    b_keys = data['option_b_ratings'].apply(tuple)
    mask = ((a_keys == target) & (b_keys == other)) | ((a_keys == other) & (b_keys == target))
    if not mask.any():
        return 0.5
    rel_data = data[mask]
    chosen_a = rel_data['response'] == 0
    a_is_target = a_keys[mask] == target
    chose_target = (chosen_a & a_is_target) | (~chosen_a & ~a_is_target)
    return float(chose_target.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.7067 (var=0.0378)
- Predicted under pi_2 (simulated): 0.4883 (var=0.0156)
- Observed on real data: 0.5133 (var=0.0354)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 5: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To robustly dissociate Tallying from WADD, we use a 5-feature design with a linear spread of validities (0.9 to 0.5). Tallying computes a simple sum of winning features for each option, completely ignoring the validities. In contrast, WADD computes a weighted sum where the validities are transformed by a non-linear parameter gamma. The trials are carefully chosen to pit the number of winning features against the validities of those features. For instance, some trials feature an option that wins on a single high-validity cue but loses on multiple lower-validity cues, leading Tallying to predict the latter and WADD (especially with gamma > 1) to predict the former. We also include several 'tie' trials where Tallying predicts equal preference (e.g., both options win on 2 features), but WADD strongly predicts the option with the higher-validity features.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times Option A is chosen on Trials 4 and 5. In these trials, both options win on exactly two features, so Tallying predicts an exact tie (choosing A 50% of the time). However, Option A wins on higher-validity features than Option B in both cases. WADD (with any gamma > 0) will consistently assign a higher value to Option A, predicting a choice probability strictly greater than 50%.

Source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    mask = ((a_str == '10100') & (b_str == '01010')) | ((a_str == '01100') & (b_str == '00011'))
    
    target_data = data[mask]
    if len(target_data) == 0:
        return 0.5
        
    return float((target_data['response'] == 0).mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.6417 (var=0.0197)
- Predicted under pi_2 (simulated): 0.5325 (var=0.0103)
- Observed on real data: 0.5400 (var=0.0176)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    
    ttb_choices = []
    for a, b in zip(a_mat, b_mat):
        choice = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                choice = 0
                break
            elif b[i] > a[i]:
                choice = 1
                break
        ttb_choices.append(choice)
        
    ttb_choices = np.array(ttb_choices)
    responses = data['response'].values
    
    valid = ttb_choices != 0.5
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(responses[valid] == ttb_choices[valid]))
```

**Observed (real) value:** 0.4408 (var=0.0085)
**Predicted under pi_3:** 0.4402 (var=0.0132)
**Predicted under pi_2:** 0.2252 (var=0.0040)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying counts the number of features where one option strictly dominates the other
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pred = np.zeros(len(data))
    tally_pred[b_wins > a_wins] = 1
    tally_pred[a_wins == b_wins] = -1 # Ignore ties for matching purposes
    
    matches = (data['response'].values == tally_pred)
    valid = (tally_pred != -1)
    
    return float(np.mean(matches[valid])) if np.any(valid) else 0.5
```

**Observed (real) value:** 0.6317 (var=0.0047)
**Predicted under pi_3:** 0.6663 (var=0.0137)
**Predicted under pi_2:** 0.8548 (var=0.0074)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, we see a clear double dissociation. In Experiments 1 and 2, the observed data strongly aligns with Tallying (Theory 2), showing that when the number of winning features is tied, subjects guess at near-chance levels (~0.51-0.54), completely ignoring the objective validities. WADD (Theory 1) incorrectly predicts a strong preference for the options with higher-validity features. However, in Experiments 3 and 4, WADD significantly outperforms Tallying. Tallying severely underpredicts the Take-The-Best match rate in Experiment 3 and overpredicts the Tallying match rate in Experiment 4. This implies that neither pure Tallying nor pure WADD is sufficient to explain the population's behavior across all contexts. The population likely consists of a mixture of strategies, or individuals dynamically switch strategies depending on the dispersion of validities or the specific choice environment.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Strategy Mixture' or 'Dual Process' theory. Instead of assuming all subjects use pure Tallying or pure WADD, the new theory should posit that a proportion of choices (or subjects) rely on a simple unweighted Tallying heuristic, while the remainder rely on a weighted compensatory strategy (like WADD) or a lexicographic strategy (like Take The Best). The model should include a mixture parameter (e.g., 'alpha') that interpolates between the choice probabilities of Tallying and WADD. This will allow the model to capture the exact chance-level behavior in tie-breaking scenarios (by leaning on the Tallying component) while still capturing the sensitivity to validities seen in other experiments."
}
```

## Usage

```json
{
  "prompt_token_count": 5221,
  "candidates_token_count": 386,
  "total_token_count": 6366
}
```
