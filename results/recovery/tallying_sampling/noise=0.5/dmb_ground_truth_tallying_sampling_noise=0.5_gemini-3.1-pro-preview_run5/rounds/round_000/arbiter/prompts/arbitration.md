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
People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This design dissociates the non-compensatory 'Take The Best' (TTB) heuristic from the fully compensatory Weighted Additive (WADD) rule. By establishing a set of validities with a clear descending order but relatively close values, we can construct 'compensatory' trials where the single best discriminating cue favors Option A, but a coalition of lower-validity cues strongly favors Option B. TTB will exclusively follow the highest-validity discriminating cue and choose A, ignoring the lower-validity cues. In contrast, WADD integrates all cues weighted by their validities, allowing the sum of the lower-validity cues to eclipse the single best cue, thus choosing B. We also include 'control' trials where both models agree, ensuring that differences in model fit are driven by the specific cue-conflict structure rather than an overall failure to capture behavior.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design consists of trials where Option A generally has a single high-validity cue in its favor, while Option B has multiple lower-validity cues. Take The Best (TTB) will consistently choose Option A across all trials because it stops at the first discriminating cue (which always favors A). In contrast, the Weighted Additive (WADD) model integrates all cues, and the coalition of multiple lower-validity cues for Option B will frequently outweigh Option A's single best cue. By computing the proportion of Option B choices (response == 1) specifically on trials where B has a greater number of positive cues than A, we perfectly isolate the compensatory nature of WADD from the non-compensatory nature of TTB. TTB's expected value will be near 0 (driven only by lapse noise), while WADD's will be significantly higher.

Source:
def metric(data: pd.DataFrame) -> float:
    b_cues = data['option_b_ratings'].apply(sum)
    a_cues = data['option_a_ratings'].apply(sum)
    mask = b_cues > a_cues
    if not mask.any():
        return 0.0
    return float(data.loc[mask, 'response'].mean())

### RESULTS
- Predicted under pi_1 (simulated): 0.1636 (var=0.0114)
- Predicted under pi_2 (simulated): 0.6681 (var=0.0277)
- Observed on real data: 0.6919 (var=0.0034)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 11: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 12: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** This design tests the compensatory nature of the Weighted Additive (WADD) rule against the non-compensatory Take The Best (TTB) heuristic by presenting choices where the single most valid discriminating cue is pitted against a coalition of multiple less valid cues. Validities are spread such that the highest validity cue is very strong (0.95), but the sum of the remaining cues is substantially larger. In the critical 'compensatory' trials, TTB will always select the option favored by the single highest-validity discriminating cue, ignoring the rest. WADD, evaluating the weighted sum of all cues, will choose the option favored by the coalition of lower-validity cues. We also include 'control' trials where both models agree (e.g., the highest validity cue and the sum of lower validity cues point to the same option) to ensure that overall task engagement is captured.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
Calculates the proportion of trials where the participant's choice aligns with the Take The Best (TTB) heuristic. TTB models will exhibit a very high match rate (close to 1.0, modulo noise), whereas WADD models, being compensatory, will frequently choose the option favored by a coalition of lower-validity cues (e.g., choosing B when A has the single best cue but B has many good cues), resulting in a significantly lower match rate.

Source:
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                total += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                total += 1
                break
    return float(matches / total) if total > 0 else 0.5

### RESULTS
- Predicted under pi_1 (simulated): 0.8435 (var=0.0127)
- Predicted under pi_2 (simulated): 0.4569 (var=0.0152)
- Observed on real data: 0.3892 (var=0.0034)

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
  "interpretation": "Across both experiments, the data overwhelmingly reject the non-compensatory Take The Best (TTB) model (THEORY 1) in favor of the compensatory Weighted Additive (WADD) model (THEORY 2). In Experiment 1, participants chose the option with more positive cues (Option B) despite Option A having the single highest-validity cue at a rate of ~69%, matching WADD's prediction (~67%) and falsifying TTB (~16%). In Experiment 2, the overall TTB match rate was only ~39%, closely matching WADD's prediction (~46%) and completely at odds with TTB's prediction (~84%). Therefore, THEORY 1 is degenerate for this dataset and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Tallying' (Equal-Weight) theory. Tallying posits that participants simply count the number of positive cues for each option and choose the one with the higher count, completely ignoring cue validities. This serves as a strong alternative compensatory benchmark to WADD: it is compensatory because multiple lower-validity cues can outvote a single high-validity cue, but it is more frugal than WADD because it does not require multiplying by or storing cardinal validities. Comparing Tallying to WADD will test whether participants actually use the provided validities or just use a simple unweighted sum."
}
```

## Usage

```json
{
  "prompt_token_count": 4642,
  "candidates_token_count": 323,
  "total_token_count": 5386
}
```
