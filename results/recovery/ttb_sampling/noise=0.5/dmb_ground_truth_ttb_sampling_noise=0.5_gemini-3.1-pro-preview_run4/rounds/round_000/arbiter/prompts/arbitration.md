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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 0, 1, 1, 1]  B=[1, 0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 7: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 9: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 10: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 0]

**Rationale:** The design contrasts Tallying (which simply counts the number of features favoring each option, ignoring validities) with Weighted Additive (WADD) (which scales feature values by their validities). By using a set of validities with a large spread between the highest and lowest values, we can create a sharp dissociation between the two models. In several critical trials, option A is superior on a larger number of low-validity features, while option B is superior on fewer but much higher-validity features. Tallying will consistently prefer the option with more positive features, while WADD will prefer the option with the higher weighted sum.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
The metric computes the proportion of choices that align with the Tallying heuristic, specifically on the subset of trials where Tallying and Weighted Additive (WADD) make strictly opposite predictions. Under the Tallying model, this proportion will be high (close to 1, subject to noise), whereas under the WADD model, it will be low (close to 0). By isolating the exact trials of disagreement, the metric maximizes the discriminative power between the two theories.

Source:
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    validities = np.array([0.95, 0.9, 0.85, 0.6, 0.55, 0.5])
    
    agreements = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        a_wadd = np.sum(a * validities)
        b_wadd = np.sum(b * validities)
        
        # Check if both models have strict preferences
        if a_wins != b_wins and a_wadd != b_wadd:
            tally_pref = 0 if a_wins > b_wins else 1
            wadd_pref = 0 if a_wadd > b_wadd else 1
            
            # Focus only on trials where Tallying and WADD disagree
            if tally_pref != wadd_pref:
                agreements.append(1.0 if row['response'] == tally_pref else 0.0)
                
    if len(agreements) == 0:
        return 0.5
    return float(np.mean(agreements))

### RESULTS
- Predicted under pi_1 (simulated): 0.8633 (var=0.0089)
- Predicted under pi_2 (simulated): 0.4407 (var=0.0546)
- Observed on real data: 0.3133 (var=0.0068)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.55, 0.5, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]

**Rationale:** This design sharply dissociates WADD from Tallying by leveraging a steep drop-off in feature validities. By pairing two highly valid features with several low-validity features, we create scenarios where Tallying (which simply counts the number of winning features) and WADD (which weights each win by its validity) make conflicting predictions. Specifically, some trials feature an option that wins on many low-validity features (favored by Tallying) against an option that wins on fewer, but highly valid features (favored by WADD). We also include trials where Tallying predicts a strict tie but WADD predicts a strong preference, ensuring a comprehensive quantitative dissociation.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
The metric computes the proportion of choices that align with the WADD model's predictions on two maximally dissociative trials (Trial 1 and Trial 3). On these trials, one option has fewer but highly valid features (favored by WADD), while the other option has more but less valid features (favored by Tallying). Because the theories make strictly opposite predictions here, WADD will score highly on this metric, while Tallying will score very poorly, yielding a large and highly significant difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract option A as a 2D array
    A = np.vstack(data['option_a_ratings'].values)
    
    # Identify Trial 1: A = [0, 0, 1, 1, 1, 0]
    # WADD prefers B (response 1), Tallying prefers A (response 0)
    is_t1 = (A[:, 0] == 0) & (A[:, 1] == 0) & (A[:, 2] == 1) & (A[:, 3] == 1)
    
    # Identify Trial 3: A = [1, 1, 0, 0, 0, 0]
    # WADD prefers A (response 0), Tallying prefers B (response 1)
    is_t3 = (A[:, 0] == 1) & (A[:, 1] == 1) & (A[:, 2] == 0) & (A[:, 3] == 0)
    
    responses = data['response'].values
    
    wadd_match = 0
    total = 0
    
    wadd_match += np.sum(responses[is_t1] == 1)
    total += np.sum(is_t1)
    
    wadd_match += np.sum(responses[is_t3] == 0)
    total += np.sum(is_t3)
    
    return float(wadd_match / total) if total > 0 else 0.5


### RESULTS
- Predicted under pi_1 (simulated): 0.1456 (var=0.0073)
- Predicted under pi_2 (simulated): 0.5719 (var=0.0907)
- Observed on real data: 0.7188 (var=0.0044)

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
  "interpretation": "Across both experiments, the observed data strongly contradicts the Tallying heuristic (THEORY 1) and aligns much better with the Weighted Additive model (THEORY 2). In Experiment 1, the metric measured the proportion of choices matching Tallying when it disagreed with WADD; the observed value was 0.31, meaning subjects predominantly chose the WADD prediction. In Experiment 2, the metric measured the proportion of choices matching WADD on maximally dissociative trials; the observed value was 0.72, again showing a strong preference for the WADD prediction over Tallying. Tallying completely fails to capture the subjects' reliance on feature validities. Therefore, THEORY 1 is degenerate and should be replaced with a stronger competitor to WADD.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose 'Take The Best' (TTB) as the new theory. Unlike Tallying (which counts unweighted wins) and WADD (which integrates all weighted features), TTB is a non-compensatory heuristic that searches through features in descending order of their validity, stopping at the first feature that discriminates between the two options and choosing the option with the higher value on that feature. This will provide a much stronger and more realistic heuristic baseline to contrast against the fully compensatory WADD model."
}
```

## Usage

```json
{
  "prompt_token_count": 4622,
  "candidates_token_count": 300,
  "total_token_count": 5595
}
```
