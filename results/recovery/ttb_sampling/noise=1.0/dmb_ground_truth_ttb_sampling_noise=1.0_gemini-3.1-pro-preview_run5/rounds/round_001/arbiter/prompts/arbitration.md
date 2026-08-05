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
Probabilistic Cue Sampling with Softmax Validities: Decision-makers evaluate options by sampling a single feature on each trial. The probability of sampling a feature is determined by a softmax function over the objective validities scaled by a sensitivity parameter gamma. The option that is superior on the sampled feature is chosen; ties result in guessing. This predicts intermediate choice proportions when validities are mixed, avoiding deterministic winner-takes-all behavior.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])

    # Compute subjective validities using softmax for sampling probabilities
    z = gamma * validities
    z = z - np.max(z)  # numerical stability
    e = np.exp(z)
    p_feat = e / np.sum(e)

    a, b = stim[0], stim[1]

    # Identify wins and ties for each feature
    a_wins = (a > b).astype(float)
    b_wins = (b > a).astype(float)
    ties = (a == b).astype(float)

    # Analytically marginalize over the sampled feature:
    # P(Choose A) = sum_i P(sample i) * P(Choose A | sample i)
    # P(Choose A | sample i) = 1 if a_i > b_i, 0 if b_i > a_i, 0.5 if a_i == b_i
    p_A = np.sum(p_feat * (a_wins + 0.5 * ties))
    p_B = np.sum(p_feat * (b_wins + 0.5 * ties))

    p_core = np.array([p_A, p_B])

    # Apply uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_2
People compare two options by computing, for each option, a weighted sum of its feature values, where each feature is weighted by its subjective validity (or importance). The option with the higher weighted sum is chosen. WADD is the compensatory benchmark against which one-reason heuristics like Take The Best are contrasted: a large deficit on a high-validity cue can be compensated by a sufficiently strong advantage on lower-validity cues, so no single feature is ever decisive on its own. Unlike Tallying, WADD uses cardinal feature magnitudes and weights them by validity, so it exploits both the sign and the size of each feature-wise comparison as well as inter-cue differences in informativeness. Unlike Equal-Weight, weights differ across features. Behavior is invariant to a shared affine rescaling across options but scales linearly with per-feature weight. When the two weighted sums are equal the model has no basis for preference and the learner must guess. Response noise enters through a softmax over the two weighted sums with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Weighted Additive rule (Payne, Bettman & Johnson
    # 1993). Stimulus is the pair of option feature vectors for the
    # current trial: array-like of shape (2, n_features), row 0 =
    # option A, row 1 = option B. Each option's score is the dot
    # product of its feature vector with the (subjective) validity
    # weights. History is ignored: the choice depends only on the
    # current stimulus.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    if w.shape[0] != n_features:
        raise ValueError(
            f"weights length {w.shape[0]} != n_features {n_features}."
        )

    # Weighted sum per option (dot product with per-feature weights).
    scores = stim @ (validities * w)

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    # Uniform lapse blended into the softmax (standard choice-with-lapse
    # formulation; epsilon=0.5 yields equiprobable options, matching
    # the paper's maximum-noise condition).
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]

**Rationale:** To quantitatively dissociate Probabilistic Cue Sampling (PCS) from the Weighted Additive rule (WADD), this design exploits the core difference in their integration mechanisms: PCS uses a softmax over validities to sample a single feature, amplifying the influence of the highest-validity cue, whereas WADD linearly integrates all features. We use four cues with linearly spaced validities. We include 'compensatory' trials where one option possesses the single highest-validity cue but the other possesses all remaining cues (PCS prefers the former, WADD strongly prefers the latter). We also include 'tie-breaker' trials where the sum of validities is equal across both options (yielding a tie in WADD), but the convexity of the softmax in PCS leads to a strong preference for the option with the most valid cue.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Probabilistic Cue Sampling (PCS) evaluates options stochastically by sampling a single cue proportional to its validity. Because the maximum sampling probability for any single cue is structurally limited by the softmax function over the validities, PCS predicts choice probabilities that are always tightly clustered around 0.5 (typically between 0.35 and 0.65). Consequently, the intra-subject variance of choices for any given trial type will consistently be high (near the theoretical maximum of 0.25).

In contrast, the Weighted Additive (WADD) model integrates all cues into a single scalar score. For most combinations of random subjective weights and inverse temperature (beta), WADD strongly favors one option over the other, leading to near-deterministic choice probabilities (close to 0 or 1), disrupted only by the uniform lapse rate (epsilon). Therefore, WADD's predicted intra-subject variance of choices will be significantly lower on average.

By calculating the empirical Bernoulli variance `p * (1 - p)` of each subject's choices for each trial type and averaging these variances, we obtain a metric that is consistently high and tightly bounded across subjects for PCS, but widely distributed and significantly lower on average for WADD. This structural guarantee ensures a massive mean gap relative to the between-subject variance, robustly discriminating the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Convert the option ratings lists into strings for safe hashing and comparison
    a_strs = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    b_strs = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Create an order-independent trial type identifier. 
    # Sorting ensures that A vs B and B vs A are mapped to the same trial type.
    trial_types = [a + '_' + b if a < b else b + '_' + a for a, b in zip(a_strs, b_strs)]
    df = data.assign(trial_type=trial_types)
    
    # Calculate the empirical choice proportion (p) for each subject and trial type.
    # Since variance p*(1-p) is symmetric, it doesn't matter which option's proportion we measure.
    p = df.groupby(['subject_id', 'trial_type'])['response'].mean()
    
    # Calculate the intra-subject variance of choices for each trial type
    var = p * (1.0 - p)
    
    # Return the mean intra-subject variance across all subjects and trial types
    return float(var.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.2280 (var=0.0001)
- Predicted under pi_2 (simulated): 0.1614 (var=0.0028)
- Observed on real data: 0.2357 (var=0.0001)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 4: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 8: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 9: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  trial 10: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 11: A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  trial 12: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** This design exploits the fundamental difference between linear integration (WADD) and softmax-based probabilistic sampling (PCS). We use five cues with linearly spaced validities: [0.9, 0.8, 0.7, 0.6, 0.5]. We construct pairs where the difference in the sum of validities is identical (e.g., +0.2, -0.2, or exactly 0). Because WADD is a linear model, it predicts identical choice probabilities across trials with the same validity difference. In contrast, PCS uses a softmax function over validities to determine sampling probabilities. Due to the strict convexity of the exponential function, the difference in sampling probabilities for two cues with a fixed validity difference (e.g., 0.9 vs 0.7 compared to 0.7 vs 0.5) is much larger at the high end of the validity scale. Moreover, for trials where the sum of validities is perfectly tied (e.g., 0.9 + 0.5 vs 0.8 + 0.6), WADD predicts exactly 50/50 guessing, whereas the convexity of the PCS softmax assigns a higher sampling probability to the extreme cues, leading PCS to systematically prefer the option with the single highest validity cue. This provides a crisp, parameter-free quantitative dissociation between the two theories.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
PCS uses a softmax over validities to determine the probability of sampling each feature. Because the validities in this design are closely spaced [0.9, 0.8, 0.7, 0.6, 0.5] and the sensitivity parameter gamma is capped at 2.0, the sampling probabilities remain relatively flat. As a result, PCS predicts that choice probabilities will be heavily compressed toward 0.5 across all trial types, making choices highly stochastic. In contrast, WADD computes a weighted sum of features. With the inverse temperature beta drawn from [0.1, 20.0], WADD frequently produces large subjective score differences, leading to highly deterministic choice probabilities (near 0 or 1) for many trial types. We can discriminate the theories by computing the variance of each subject's empirical choice proportions across the 12 trial types. PCS predicts a low variance (dominated by binomial noise around p=0.5), while WADD predicts a significantly higher variance due to its capacity for deterministic responding.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    df = data.copy()
    # Create hashable trial identifiers safely using list comprehensions
    df['trial_id'] = df['option_a_ratings'].apply(lambda x: "".join([str(int(v)) for v in x])) + \
                     "_" + \
                     df['option_b_ratings'].apply(lambda x: "".join([str(int(v)) for v in x]))
    
    subj_vars = []
    for subj, subj_df in df.groupby('subject_id'):
        # Proportion of choosing A (response == 0) for each trial type
        p_A = subj_df.groupby('trial_id')['response'].apply(lambda x: np.mean(x == 0))
        # Variance of these proportions across the 12 trial types
        subj_vars.append(p_A.var(ddof=1))
        
    return float(np.mean(subj_vars))

### RESULTS
- Predicted under pi_3 (simulated): 0.0334 (var=0.0002)
- Predicted under pi_2 (simulated): 0.0987 (var=0.0032)
- Observed on real data: 0.0341 (var=0.0001)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    tally_consistent = 0
    total_incongruent = 0
    
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        resp = row['response']
        
        # Incongruent trial 1: Option A has fewer but higher-validity features
        if a == [1, 1, 0, 0, 0] and b == [0, 0, 1, 1, 1]:
            total_incongruent += 1
            if resp == 1:  # Tallying prefers B (3 features > 2 features)
                tally_consistent += 1
        
        # Incongruent trial 4: Option B has fewer but higher-validity features
        elif a == [0, 0, 1, 1, 1] and b == [1, 1, 0, 0, 0]:
            total_incongruent += 1
            if resp == 0:  # Tallying prefers A (3 features > 2 features)
                tally_consistent += 1
                
    if total_incongruent == 0:
        return 0.5
    return tally_consistent / total_incongruent
```

**Observed (real) value:** 0.5067 (var=0.0125)
**Predicted under pi_3:** 0.4892 (var=0.0113)
**Predicted under pi_2:** 0.4117 (var=0.1164)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where one option has the two highest validity features [1, 1, 0, 0, 0]
    # and the other has the three lowest validity features [0, 0, 1, 1, 1].
    is_t1 = (data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))) & (data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)))
    is_t2 = (data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))) & (data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)))
    
    wadd_choices = 0
    total = 0
    
    if is_t1.any():
        wadd_choices += (data.loc[is_t1, 'response'] == 0).sum()
        total += is_t1.sum()
        
    if is_t2.any():
        wadd_choices += (data.loc[is_t2, 'response'] == 1).sum()
        total += is_t2.sum()
        
    return float(wadd_choices / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.4800 (var=0.0099)
**Predicted under pi_3:** 0.4967 (var=0.0119)
**Predicted under pi_2:** 0.5533 (var=0.0817)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all four experiments, Theory 1 (Probabilistic Cue Sampling) consistently outperforms Theory 2 (Weighted Additive). In Experiment 1, Theory 1 accurately predicts the high intra-subject variance (observed 0.2357, predicted 0.2280), whereas WADD predicts choices that are too deterministic (0.1614). In Experiment 2, Theory 1 captures the low variance of choice proportions across trial types (observed 0.0341, predicted 0.0334), while WADD predicts a much higher variance (0.0987) due to its sensitivity to linear score differences. Experiments 3 and 4 further confirm that Theory 1's stochastic sampling mechanism closely matches observed choice proportions, whereas WADD consistently overestimates the determinism of choices. Theory 2's deterministic linear integration fails to capture the high degree of stochasticity and compression toward 0.5 seen in the data.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the WADD model with a 'Stochastic Take-The-Best' (Lexicographic) theory. In this theory, decision-makers search through features sequentially in descending order of their objective validities. However, the search process is noisy: at each step, there is a probability of prematurely terminating the search and guessing, or a probability of misperceiving the cue values (execution error). If a cue discriminates, the decision-maker chooses the superior option with high probability. This provides a strongly contrasting algorithmic mechanism to Theory 1 (sequential strict order vs. single-shot softmax sampling) while still structurally predicting the high stochasticity and intermediate choice proportions observed in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 6213,
  "candidates_token_count": 381,
  "total_token_count": 7425
}
```
