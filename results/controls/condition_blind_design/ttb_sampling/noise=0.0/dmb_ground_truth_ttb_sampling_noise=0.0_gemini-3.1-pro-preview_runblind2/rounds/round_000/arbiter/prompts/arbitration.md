# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_1
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


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.95, 0.82, 0.84, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 2: A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  trial 5: A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  trial 7: A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  trial 9: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 10: A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  trial 13: A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  trial 14: A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 15: A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  trial 16: A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
By isolating trials where Tallying encounters a tie (i.e., both options win on an equal number of features), we can cleanly separate the two theories. Tallying predicts random guessing (0.5) on these trials regardless of its parameters. In contrast, WADD uses the specific validities to break the tie, consistently favoring the option with higher validity-weighted features. Measuring the proportion of choices that align with WADD's prediction on these specific tie trials yields exactly 0.5 for Tallying and strictly > 0.5 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.82, 0.84, 0.55])
    
    match_count = 0
    tie_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Identify trials where Tallying predicts a tie
        if a_wins == b_wins and a_wins > 0:
            tie_count += 1
            score_a = np.sum(a * val)
            score_b = np.sum(b * val)
            
            # Check if response aligns with WADD's strict preference
            if score_a > score_b and row['response'] == 0:
                match_count += 1
            elif score_b > score_a and row['response'] == 1:
                match_count += 1
                
    if tie_count == 0:
        return 0.5
        
    return float(match_count / tie_count)

### RESULTS
- Predicted under pi_1 (simulated): 0.5022 (var=0.0117)
- Predicted under pi_2 (simulated): 0.5722 (var=0.0442)
- Observed on real data: 0.8422 (var=0.0120)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.78, 0.88, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 2: A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  trial 3: A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  trial 4: A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  trial 5: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 6: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 7: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 8: A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  trial 9: A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  trial 10: A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  trial 11: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 12: A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  trial 13: A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  trial 14: A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  trial 15: A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  trial 16: A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
The metric contrasts the probability of choosing option B on two specific trial types where Tallying and WADD systematically diverge. On both Trial 4 and Trial 9, each option has exactly one unique winning feature, so Tallying always registers a tie and predicts a 50/50 choice (difference = 0). However, WADD weighs features by their validities. In Trial 4, B's unique advantage (0.78) outweighs A's (0.55), leading WADD to favor B. In Trial 9, A's unique advantage (0.95) heavily outweighs B's (0.55), leading WADD to favor A. Thus, WADD predicts a positive difference (P(B|Trial 4) - P(B|Trial 9) > 0), whereas Tallying predicts exactly zero.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Trial 4: A=[0, 0, 1, 1], B=[0, 1, 1, 0]
    # A wins on feature 4 (validity 0.55), B wins on feature 2 (validity 0.78)
    mask4 = (a_str == '0011') & (b_str == '0110')
    
    # Trial 9: A=[1, 0, 1, 0], B=[0, 0, 1, 1]
    # A wins on feature 1 (validity 0.95), B wins on feature 4 (validity 0.55)
    mask9 = (a_str == '1010') & (b_str == '0011')
    
    p_b_4 = data.loc[mask4, 'response'].mean()
    p_b_9 = data.loc[mask9, 'response'].mean()
    
    if pd.isna(p_b_4):
        p_b_4 = 0.5
    if pd.isna(p_b_9):
        p_b_9 = 0.5
        
    return float(p_b_4 - p_b_9)

### RESULTS
- Predicted under pi_1 (simulated): -0.0433 (var=0.0742)
- Predicted under pi_2 (simulated): 0.2200 (var=0.1860)
- Observed on real data: 0.7000 (var=0.0889)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
(no other experiments)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across both experiments, the observed data strongly contradicts Theory 1 (Tallying), which predicts random guessing on trials where the number of winning features is equal. Instead, subjects systematically favor the option with higher-validity features, as shown by the high observed metrics (0.84 in Exp 1, 0.70 in Exp 2). While Theory 2 (WADD) predicts the correct qualitative direction (values > 0.5 and > 0, respectively), the observed effects are much stronger than WADD's simulated predictions. This suggests that subjects are relying heavily on the cue validities, potentially in a non-compensatory, lexicographic manner rather than a noisy weighted sum. Theory 1 (Tallying) is fundamentally unable to explain this validity-driven behavior and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose the 'Take-The-Best' (TTB) heuristic as a new theory. In TTB, decision-makers rank features by their validities and compare options lexicographically: they look at the feature with the highest validity first, and if one option is strictly better on that feature, they choose it. If there is a tie, they move to the second most valid feature, and so on. This differs from Tallying (which ignores validities entirely) and WADD (which computes a compensatory weighted sum). TTB's deterministic reliance on the highest-validity discriminating cue is likely to produce the strong directional effects observed in the data."
}
```

## Usage

```json
{
  "prompt_token_count": 4769,
  "candidates_token_count": 342,
  "total_token_count": 5850
}
```
