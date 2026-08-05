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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[0, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** This design systematically dissociates Tallying from the Weighted Additive (WADD) rule by pitting a larger number of low-validity features against a smaller number of high-validity features. Tallying simply counts the number of features on which an option is superior, ignoring the validities entirely. WADD weights each feature by its subjective validity. By selecting a steep drop-off in validities (e.g., 0.95 and 0.90 vs. 0.60, 0.55, 0.50), we can create trials where one option wins on more features (favored by Tallying) but the other option has a higher validity-weighted sum (favored by WADD). We also include trials where Tallying predicts a tie (equal number of winning features) while WADD strongly prefers one option, as well as control trials where both models agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Tallying heuristic on the specific trials where Tallying and WADD make diametrically opposed predictions. In Trial 1 and Trial 2, one option has a higher count of winning features (favored by Tallying), while the other option has a higher validity-weighted sum (favored by WADD). Tallying models will score close to 1.0, whereas WADD models will score close to 0.0, maximizing discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    tallying_choices = 0
    total_relevant = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            if resp == 0:
                tallying_choices += 1
            total_relevant += 1
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            if resp == 1:
                tallying_choices += 1
            total_relevant += 1
    if total_relevant == 0:
        return 0.5
    return tallying_choices / total_relevant

### RESULTS
- Predicted under pi_1 (simulated): 0.8517 (var=0.0133)
- Predicted under pi_2 (simulated): 0.4967 (var=0.0864)
- Observed on real data: 0.3400 (var=0.0108)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This design systematically dissociates the Weighted Additive (WADD) theory from the Tallying heuristic by exploiting the difference between validity-weighted sums and unweighted feature-win counts. We use a set of 5 features with a steep drop-off in validities. In some trials, one option wins on more features (favored by Tallying) but the other option has a higher validity-weighted sum (favored by WADD). We also include trials where Tallying predicts a tie (equal number of winning features) while WADD strongly prefers one option, as well as trials where the models agree to ensure a balanced design. This allows for a robust quantitative dissociation between the two choice mechanisms.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In the specific trial where option A has the two most valid cues and option B has the three least valid cues, the WADD theory predicts a preference for option A because the sum of the highest validities outweighs the sum of the lowest. Conversely, the Tallying heuristic predicts a preference for option B because it has more positive cues in total. Measuring the proportion of times option B is chosen on this specific trial robustly discriminates the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify the specific trial where WADD and Tallying strongly disagree.
    # Trial: A=[1, 1, 0, 0, 0] vs B=[0, 0, 1, 1, 1]
    # WADD prefers A (validity sum 1.8 > 1.7) while Tallying prefers B (tally 3 > 2).
    mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    if mask.sum() == 0:
        return 0.5
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.8683 (var=0.0120)
- Predicted under pi_2 (simulated): 0.5483 (var=0.0811)
- Observed on real data: 0.3567 (var=0.0242)

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
  "interpretation": "Across both experiments, the observed data consistently contradicts the Tallying heuristic (Theory 1) and leans towards a compensatory mechanism like WADD (Theory 2). In both experiments, subjects were presented with choices where Tallying favored an option with more positive cues, while WADD favored an option with fewer but highly valid cues. The observed metrics (around 0.34 - 0.35) indicate that subjects chose the Tallying-favored option only about 35% of the time, meaning they preferred the WADD-favored option about 65% of the time. However, the current implementation of Theory 2 (WADD) predicted a metric around 0.50. This is because the implementation incorrectly introduces a free `weights` parameter array that multiplies the validities, effectively randomizing the feature weights instead of directly using the instructed validities. If WADD correctly used the provided validities, it would correctly predict the preference for the high-validity options.",
  "verdict": "new_model",
  "target_theory_idx": 2,
  "recommendation": "Revise the `predict` function to remove the independent `weights` array parameter. The weighted sum should be computed directly using the provided `validities`. To account for subjective distortions of probability/validity, you may introduce a single scalar parameter `gamma` to exponentiate the validities (i.e., subjective_weight = validities ** gamma), but do not use an unconstrained array of weights that ignores the instructed validities. Keep the softmax `beta` and lapse `epsilon` parameters."
}
```

## Usage

```json
{
  "prompt_token_count": 4262,
  "candidates_token_count": 348,
  "total_token_count": 5572
}
```
