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
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** This design tests the quantitative dissociation between Tallying (which counts the number of features won and ignores validities) and WADD (which computes a validity-weighted sum). By using five features with a mix of high and low validities, we can construct trials where one option wins on a greater number of low-validity features, while the other wins on fewer high-validity features. In these crucial trials, Tallying predicts the option with more winning features, whereas WADD predicts the option with the higher weighted sum. Additional control trials where both models agree or where Tallying predicts a tie (but WADD has a strict preference) ensure the full parameter space of both models can be identified.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the key dissociation trials where Tallying and WADD make perfectly opposite predictions. In trials where A=[0,0,1,1,1] and B=[1,1,0,0,0], Tallying prefers A because it wins on 3 features vs 2, while WADD prefers B because the validity-weighted sum is higher for the fewer but more valid features (0.95+0.85 = 1.80 vs 0.65+0.55+0.5 = 1.70). The metric calculates the proportion of choices that align with Tallying's prediction on these specific trials. Under Tallying, this proportion will be high (>0.5), whereas under WADD it will be low (<0.5), creating a massive mean gap with minimal variance.

Source:
def metric(data: pd.DataFrame) -> float:
    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)) & data['option_b_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    t1_data = data[is_t1]
    t2_data = data[is_t2]
    
    t1_match = (t1_data['response'] == 0).sum()
    t2_match = (t2_data['response'] == 1).sum()
    
    total = len(t1_data) + len(t2_data)
    
    if total == 0:
        return 0.5
    return float(t1_match + t2_match) / total


### RESULTS
- Predicted under pi_1 (simulated): 0.8531 (var=0.0108)
- Predicted under pi_2 (simulated): 0.3944 (var=0.0773)
- Observed on real data: 0.1062 (var=0.0063)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]

**Rationale:** This design tests the quantitative dissociation between WADD (which computes a validity-weighted sum) and Tallying (which counts the number of features won, ignoring validities). By using six features with a mix of high and low validities, we construct trials where one option wins on a greater number of low-validity features, while the other wins on fewer high-validity features. In these crucial trials, Tallying predicts the option with more winning features, whereas WADD predicts the option with the higher weighted sum. Additional trials where Tallying predicts a tie but WADD has a strict preference, or where both models agree, ensure the full parameter space of both models can be identified.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Tallying heuristic specifically on the critical trials (Trials 1, 2, and 3). In these trials, one option possesses 2 high-validity features (and 0 low-validity features), while the competing option possesses 3 low-validity features (and 0 high-validity features). Tallying strictly prefers the option with 3 features, whereas WADD correctly weighs the validities and strictly prefers the option with 2 high-validity features. This produces a stark divergence: the metric will be near 1.0 for Tallying and near 0.0 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    tally_choices = 0
    total_critical = 0
    for _, row in data.iterrows():
        a = list(row['option_a_ratings'])
        b = list(row['option_b_ratings'])
        
        sum_a = sum(a)
        sum_b = sum(b)
        sum_a_top = sum(a[:3])
        sum_b_top = sum(b[:3])
        
        # Check if this is a critical trial: one option has exactly 2 high-validity features, 
        # while the other has exactly 3 low-validity features.
        if sum_a == 2 and sum_a_top == 2 and sum_b == 3 and sum_b_top == 0:
            total_critical += 1
            if row['response'] == 1:
                tally_choices += 1
        elif sum_b == 2 and sum_b_top == 2 and sum_a == 3 and sum_a_top == 0:
            total_critical += 1
            if row['response'] == 0:
                tally_choices += 1
                
    return tally_choices / total_critical if total_critical > 0 else 0.5

### RESULTS
- Predicted under pi_1 (simulated): 0.8589 (var=0.0135)
- Predicted under pi_2 (simulated): 0.5250 (var=0.0518)
- Observed on real data: 0.2022 (var=0.0207)

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
  "interpretation": "Across both experiments, the observed data strongly rejects Tallying (Theory 1). In these critical trials, Tallying predicts a strong preference for the option with more overall features (yielding metric values ~0.85). However, the observed metrics are ~0.10 and ~0.20, indicating that subjects overwhelmingly chose the option with fewer, but higher-validity, features. While WADD (Theory 2) correctly predicts the direction of this preference in principle, its simulated metric values (0.39 and 0.52) are much closer to 0.5 because the weighted sum differences are relatively small, leading to weak preferences under the softmax choice rule. The fact that human subjects show such an extreme preference (e.g., 90% in Exp 1) for the option winning on the most valid features suggests they are using a non-compensatory strategy rather than computing a weighted sum. Specifically, a lexicographic heuristic like Take-The-Best (TTB) would predict a near 0.0 metric on these trials because it decides entirely based on the single most valid discriminating cue.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace Tallying with the 'Take-The-Best' (TTB) heuristic. The new theory should propose that subjects search through features in descending order of their validity. They stop at the very first feature that discriminates between the two options (i.e., one option has a 1 and the other has a 0) and deterministically choose the option that wins on that cue, ignoring all lower-validity features entirely. If all features tie, they guess. Response noise can be modeled as a simple lapse rate (epsilon) that results in a random guess."
}
```

## Usage

```json
{
  "prompt_token_count": 4505,
  "candidates_token_count": 391,
  "total_token_count": 6022
}
```
