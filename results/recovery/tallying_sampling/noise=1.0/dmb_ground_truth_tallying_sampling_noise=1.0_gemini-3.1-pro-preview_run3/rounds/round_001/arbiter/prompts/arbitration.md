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
Probabilistic Cue Selection (Random Cue) posits that decision-makers do not deterministically use the most valid cue (like Take-The-Best) nor do they integrate all cues simultaneously (like WADD). Instead, on each trial, they sample a single cue with a probability proportional to its subjective validity. They then choose the option favored by that sampled cue, guessing uniformly if the sampled cue ties. This single-cue sampling process naturally generates probabilistic choices across trials, producing choice shares near 0.50 for conflict trials where different cues favor different options, without relying on extreme softmax noise.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # Subjective validities used as sampling weights
    subj_weights = val * w
    sum_w = np.sum(subj_weights)
    
    if sum_w <= 1e-9:
        p_core = np.array([0.5, 0.5])
    else:
        p_cue = subj_weights / sum_w
        a, b = stim[0], stim[1]
        
        p_a = 0.0
        for j in range(n_features):
            if a[j] > b[j]:
                p_a += p_cue[j]
            elif a[j] == b[j]:
                p_a += p_cue[j] * 0.5
                
        p_core = np.array([p_a, 1.0 - p_a])
        
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


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
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 2: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 4: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 5: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 6: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 7: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 8: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 9: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 10: A=[0, 1, 0, 0]  B=[1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Probabilistic Cue Selection (Random Cue) from the Weighted Additive rule (WADD), we exploit their different mappings from cue validities to choice probabilities. WADD computes a weighted sum of features for each option and passes the difference through a softmax, predicting a sigmoidal relationship between the score difference and choice probability. In contrast, Random Cue samples a single cue proportional to its validity and chooses the favored option, predicting that choice probabilities scale strictly linearly with the proportion of validities favoring an option (ignoring extreme softmax noise). By carefully designing trials where the difference in weighted sums varies independently of the total validities involved (e.g., comparing trials with few vs. many conflicting cues), we can observe whether choice probabilities follow a linear mixture of validities (Random Cue) or a deterministic integration with softmax noise (WADD).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
To definitively discriminate between WADD and Random Cue, we can examine the within-subject choice consistency on 'easy' trials where one option is supported by 3 cues and the other by 1 (Trials 5, 6, and 9). 

In WADD, the 3v1 cue advantage produces a very large difference in weighted sums. When passed through the softmax function, this pushes the underlying choice probability almost entirely to 1.0. Thus, WADD predicts that subjects will be highly consistent in their choices across repetitions of these trials, with any deviations driven almost exclusively by the uniform lapse rate (epsilon).

In contrast, Random Cue generates choices by probabilistically sampling a single cue on each trial. Even when 3 cues favor Option A, the 1 opposing cue still has a substantial probability (roughly 20-25%) of being sampled. Therefore, Random Cue predicts inherently probabilistic behavior even before any lapse rate is applied, leading to significantly lower choice consistency across repetitions. 

By computing the mean choice consistency (the maximum of the choice proportion for A or B) specifically on these 3v1 trials, we capture a large, stable gap between the theories that minimizes cross-subject variance caused by unobservable cue weights.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    data = data.copy()
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['chose_a'] = (data['response'] == 0).astype(float)
    
    # We focus on the '3v1' trials where WADD predicts near-deterministic choices
    target_trials = ['0111', '1110', '1011']
    df_target = data[data['a_str'].isin(target_trials)]
    
    if df_target.empty:
        return 0.5
        
    # Calculate the proportion of A choices per subject and trial type
    grouped = df_target.groupby(['subject_id', 'a_str'])['chose_a'].mean()
    
    # Consistency is how far the choice proportion is from 0.5 (i.e., max(P(A), P(B)))
    consistency = np.maximum(grouped, 1.0 - grouped)
    
    return float(consistency.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.7252 (var=0.0057)
- Predicted under pi_2 (simulated): 0.8207 (var=0.0112)
- Observed on real data: 0.6489 (var=0.0031)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 11: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive rule (WADD) from Probabilistic Cue Selection (Random Cue), we exploit the functional form of their choice probabilities. Random Cue's probability of choosing an option is strictly a linear function of the difference in weighted feature sums (because P(A) = 0.5 + 0.5 * ScoreDiff / TotalValidity). In contrast, WADD passes this same score difference through a softmax function, predicting a sigmoidal (S-shaped) relationship. By presenting a series of trials that systematically vary the score difference from zero to large values, we can evaluate whether the choice proportions follow a linear mixture (Random Cue) or a deterministic integration with softmax noise (WADD).

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Rather than subtracting hard trials (which inflates the between-subject variance of the metric due to the sum of variances), we isolate the absolute accuracy strictly on the 'easiest' trials (Trials 7, 9, and 10) where the objective score difference is large (>= 0.65). Random Cue strictly limits choice probabilities to a linear mixture of the cues, meaning that even on these 'easy' trials, the accuracy remains relatively low (around ~63%). In contrast, WADD uses a softmax function over the score differences, heavily amplifying the choice probability for the better option on these specific trials (predicting ~90% accuracy). This produces a massive gap between the theories' mean predictions while keeping the between-subject variance tightly bounded.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Objective validities used in the experiment
    v = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    
    # Reconstruct the feature matrices
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Calculate objective scores and differences
    score_a = a_ratings @ v
    score_b = b_ratings @ v
    diff = score_a - score_b
    
    # Select "easy" trials with a large score difference (abs(diff) >= 0.65)
    # In this design, this cleanly isolates Trials 7, 9, and 10.
    mask = np.abs(diff) >= 0.65
    
    if not np.any(mask):
        return 0.5
        
    # Determine if the subject chose the option with the higher objective score
    is_correct = ((diff > 0) & (data['response'] == 0)) | ((diff < 0) & (data['response'] == 1))
    
    # Return the simple accuracy on these easy trials
    return float(is_correct[mask].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.6092 (var=0.0121)
- Predicted under pi_2 (simulated): 0.7192 (var=0.0279)
- Observed on real data: 0.5233 (var=0.0124)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def get_cue1_winner(row):
        if row['option_a_ratings'][0] > row['option_b_ratings'][0]:
            return 0
        elif row['option_b_ratings'][0] > row['option_a_ratings'][0]:
            return 1
        return -1
    
    cue1_winners = data.apply(get_cue1_winner, axis=1)
    valid_trials = cue1_winners != -1
    if not valid_trials.any():
        return 0.5
    
    return np.mean(data.loc[valid_trials, 'response'] == cue1_winners[valid_trials])
```

**Observed (real) value:** 0.5071 (var=0.0028)
**Predicted under pi_3:** 0.4753 (var=0.0043)
**Predicted under pi_2:** 0.4631 (var=0.0165)

### Experiment 4
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    conflict_trials = 0
    
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        if a == (1, 0, 0, 0) and b == (0, 1, 1, 1):
            if resp == 0: ttb_matches += 1
            conflict_trials += 1
        elif a == (0, 1, 1, 1) and b == (1, 0, 0, 0):
            if resp == 1: ttb_matches += 1
            conflict_trials += 1
        elif a == (0, 1, 0, 0) and b == (0, 0, 1, 1):
            if resp == 0: ttb_matches += 1
            conflict_trials += 1
        elif a == (0, 0, 1, 1) and b == (0, 1, 0, 0):
            if resp == 1: ttb_matches += 1
            conflict_trials += 1
            
    if conflict_trials == 0:
        return 0.5
    return float(ttb_matches / conflict_trials)
```

**Observed (real) value:** 0.4822 (var=0.0067)
**Predicted under pi_3:** 0.4017 (var=0.0165)
**Predicted under pi_2:** 0.3089 (var=0.0415)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, Theory 1 (Probabilistic Cue Selection / Random Cue) consistently outperforms Theory 2 (WADD). The observed data reveals surprisingly low consistency and accuracy (often hovering near 0.50 - 0.65), which heavily penalizes WADD's deterministic, validity-weighted integration that predicts much higher accuracy and consistency on 'easy' trials. While Theory 1 captures the probabilistic nature of the choices better, it still slightly overpredicts performance in Experiments 1 and 2, and underpredicts in Experiments 3 and 4. Theory 2's assumption of compensatory, validity-weighted integration is clearly failing to capture the highly noisy or non-compensatory behavior of the subjects. Therefore, Theory 2 should be replaced with a fundamentally different approach.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose an 'Equal Weights' (Tallying) theory. In this theory, subjects ignore the provided cue validities entirely and simply count the number of positive features for each option. They choose the option with the higher tally, and guess randomly if there is a tie. This bounded-rationality approach would naturally explain the very low choice consistency on trials where the validity-weighted score strongly favors one option but the raw tally of cues is close or tied. It provides a strong contrast to both the single-cue sampling of Theory 1 and the complex weighted integration of the old Theory 2."
}
```

## Usage

```json
{
  "prompt_token_count": 5965,
  "candidates_token_count": 322,
  "total_token_count": 7017
}
```
