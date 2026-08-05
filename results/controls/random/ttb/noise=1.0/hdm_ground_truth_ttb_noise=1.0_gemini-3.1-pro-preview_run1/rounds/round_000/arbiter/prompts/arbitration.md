# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Heuristic Decision Making domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of integer expert ratings in [0, rating_max]. Every experiment fixes its own feature count (via `validities` length), per-expert validities, and `rating_max`; all three are LLM-proposed. Choosing rating_max=1 gives binary ratings; rating_max>1 gives cardinal ratings whose magnitudes carry information beyond just sign. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

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
    return int(np.argmax(probabilities))


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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=3):** [0.9, 0.7, 0.6]

**rating_max:** 5

**Trial pairs (n=6):**
  trial 1: A=[1, 4, 4]  B=[5, 3, 3]
  trial 2: A=[5, 0, 0]  B=[2, 1, 1]
  trial 3: A=[2, 5, 0]  B=[3, 1, 1]
  trial 4: A=[0, 4, 4]  B=[2, 3, 3]
  trial 5: A=[0, 5, 5]  B=[3, 4, 4]
  trial 6: A=[4, 1, 0]  B=[1, 2, 2]

**Rationale:** To dissociate Tallying from WADD, we use cardinal features (rating_max = 5) and vary both the magnitude of feature differences and the validity of the features. Tallying makes decisions based purely on the count of winning features, ignoring the size of the differences and their validities. WADD, in contrast, computes a weighted sum that is sensitive to both validities and magnitudes. By setting up trials where one option wins on more features (favored by Tallying) but by very small margins on low-validity cues, while the other option wins on fewer features but by large margins and/or on high-validity cues (favored by WADD), we can perfectly separate the predictions of the two models.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental trials are specifically designed so that Tallying and WADD consistently predict opposite choices. By calculating the proportion of choices that align with the Tallying heuristic's predictions (i.e., choosing the option that wins on the majority of features, regardless of magnitude), we can perfectly discriminate between the two models. Data simulated under Tallying will yield a metric value close to 1.0 (modulo lapse rate), whereas data simulated under WADD will yield a value closer to 0.0, resulting in a massive and highly significant gap.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    tally_pred = (b_wins > a_wins).astype(int)
    
    return float((data['response'] == tally_pred).mean())

### RESULTS
- Predicted under pi_1 (simulated): 1.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.3067 (var=0.1115)
- Observed on real data: 0.4917 (var=0.0021)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=3):** [0.9, 0.7, 0.6]

**rating_max:** 10

**Trial pairs (n=6):**
  trial 1: A=[2, 6, 6]  B=[10, 5, 5]
  trial 2: A=[10, 2, 2]  B=[8, 3, 3]
  trial 3: A=[9, 5, 4]  B=[1, 5, 8]
  trial 4: A=[5, 5, 5]  B=[0, 8, 8]
  trial 5: A=[1, 9, 9]  B=[8, 8, 8]
  trial 6: A=[9, 10, 1]  B=[10, 1, 9]

**Rationale:** This design perfectly dissociates the Weighted Additive (WADD) model from Tallying by manipulating the cardinal magnitudes of feature differences and their validities. Tallying relies exclusively on the sign of feature-wise comparisons, ignoring both the size of the difference and the validity of the features. WADD computes a compensatory weighted sum sensitive to both validities and magnitudes. We use a 10-point scale (rating_max=10) and three features with validities [0.9, 0.7, 0.6]. The trials are crafted such that in many pairs, one option wins on a majority of features (thus preferred by Tallying) but by very small margins and/or on lower-validity features, while the other option wins on a minority of features (thus losing the tally) but by massive margins on high-validity features (thus preferred by WADD). Other trials introduce ties for Tallying where WADD has a strong preference, ensuring a robust quantitative dissociation.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials 1 and 5, where the predictions of Tallying and WADD are diametrically opposed and highly robust to parameter variations. In these trials, option A wins on a majority of features (features 2 and 3) but by small margins, leading Tallying to deterministically prefer A. Conversely, option B wins on the single most valid feature (feature 1) by a massive margin, which overwhelms the small deficits on the other features in a compensatory model, leading WADD to strongly prefer B. The metric calculates the proportion of choices for option B across these specific trials. It will yield a value near 0 for Tallying and near 1 for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    wadd_match = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        resp = row['response']
        # Trial 1: A=[2, 6, 6], B=[10, 5, 5]
        if a[0] == 2:
            wadd_match.append(1 if resp == 1 else 0)
        # Trial 5: A=[1, 9, 9], B=[8, 8, 8]
        elif a[0] == 1:
            wadd_match.append(1 if resp == 1 else 0)
    if not wadd_match:
        return 0.5
    return float(np.mean(wadd_match))

### RESULTS
- Predicted under pi_1 (simulated): 0.0000 (var=0.0000)
- Predicted under pi_2 (simulated): 0.8900 (var=0.0829)
- Observed on real data: 0.5000 (var=0.0073)

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
  "interpretation": "Both Tallying (pi_1) and Weighted Additive (pi_2) fail to capture the observed data. In Experiment 1, Tallying predicted a near 1.0 match and WADD predicted ~0.3, but the observed value was ~0.5. In Experiment 2, WADD predicted ~0.89 and Tallying 0.0, but the observed value was exactly 0.5. The consistent 0.5 observed values across these highly discriminative trials suggest that subjects are not using either purely compensatory weighting or unweighted tallying. Instead, they likely rely on a lexicographic heuristic like Take The Best (TTB), which decides based entirely on the single most valid discriminating feature, completely ignoring the rest.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose the 'Take The Best' (TTB) heuristic. Unlike Tallying (which counts wins equally) and WADD (which computes a compensatory weighted sum), TTB searches through cues in descending order of their validity. It stops at the first cue that strictly discriminates between the two options (i.e., where one option has a higher value than the other) and chooses the option that wins on that cue, completely ignoring all lower-validity cues regardless of their magnitudes."
}
```

## Usage

```json
{
  "prompt_token_count": 3970,
  "candidates_token_count": 297,
  "total_token_count": 5213
}
```
