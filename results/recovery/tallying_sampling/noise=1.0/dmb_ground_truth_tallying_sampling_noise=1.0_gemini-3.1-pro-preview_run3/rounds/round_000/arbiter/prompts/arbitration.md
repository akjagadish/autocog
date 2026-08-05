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

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 9: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 10: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from the Weighted Additive rule (WADD), we must exploit their fundamental difference: TTB is a non-compensatory 'one-reason' heuristic, while WADD is fully compensatory. We use five features with linearly decreasing validities. The critical trials present a 'compensatory' conflict: one option is endorsed by the single most valid cue, but the competing option is endorsed by several less valid cues whose combined weight exceeds that of the single best cue. TTB will deterministically choose the option with the highest valid cue, entirely ignoring the others. WADD, conversely, will integrate all cues and choose the option with the greater weighted sum, provided the subjective weights do not overly distort the validities. By varying which specific valid cues are in conflict (e.g., Cue 1 vs Cues 2,3,4,5 or Cues 1,3 vs Cues 2,4,5), we can map out the precise compensatory threshold for the subject, which WADD predicts will be a smooth function of the feature sums, while TTB predicts a step function strictly dependent on the top discriminating cue.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
In all chosen trial pairs, the first cue (which has the highest validity) strictly discriminates between Option A and Option B. Take The Best (TTB) will deterministically choose the option endorsed by this top cue, ignoring all other cues. In contrast, the Weighted Additive rule (WADD) integrates all cues, and since the sum of the less valid cues often exceeds the top cue in these specific compensatory trials, WADD will frequently choose the option that lacks the top cue. Therefore, the proportion of choices aligning with the top cue will be near 1.0 for TTB, but significantly lower for WADD.

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 0.8364 (var=0.0110)
- Predicted under pi_2 (simulated): 0.4631 (var=0.0165)
- Observed on real data: 0.5071 (var=0.0028)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 4: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 5: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 7: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 8: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Rationale:** To strongly dissociate the compensatory Weighted Additive (WADD) rule from the non-compensatory Take The Best (TTB) heuristic, we use a 4-feature environment where the validities are spread such that the most valid cue can be outweighed by a combination of less valid cues. We present a mix of 'conflict' trials where TTB and WADD make opposite predictions (e.g., Option A is favored by the single best cue, while Option B is favored by the sum of the remaining cues) and 'alignment' trials where both models agree. By observing choices across these specific conflicts, we can quantitatively identify whether subjects are integrating all evidence weighted by validity (WADD) or stopping at the first discriminating cue (TTB).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Take The Best (TTB) heuristic specifically on 'conflict' trials. In these trials (e.g., A=[1,0,0,0] vs B=[0,1,1,1]), the single most valid discriminating cue points to one option, but the sum of the less valid cues points to the other. TTB will consistently choose the option favored by the single best cue, resulting in a high metric value. Conversely, the Weighted Additive (WADD) model integrates all cues, allowing the numerous lesser cues to outweigh the single best cue, resulting in a low metric value.

Source:
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

### RESULTS
- Predicted under pi_1 (simulated): 0.8200 (var=0.0128)
- Predicted under pi_2 (simulated): 0.3089 (var=0.0415)
- Observed on real data: 0.4822 (var=0.0067)

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
  "interpretation": "Across both experiments, the observed metrics (around 0.50) fall roughly halfway between the strong non-compensatory predictions of Take The Best (pi_1, ~0.82-0.83) and the compensatory predictions of Weighted Additive (pi_2, ~0.31-0.46). The extremely low between-subject variance indicates that subjects are highly consistent in this ~0.50 behavior, rather than being a mix of extreme TTB and WADD users. Neither model perfectly captures this. TTB strongly overpredicts the choice share of the top cue, while WADD underpredicts it (expecting the sum of lower cues to dominate more often than it does). This suggests subjects are not strictly deterministic in either direction, or they are using a probabilistic cue-selection strategy.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Probabilistic Cue Selection' (or Random Cue) theory. Instead of strictly ordering cues by validity (as in TTB) or integrating all of them (as in WADD), this theory posits that decision-makers sample a single cue on each trial with a probability proportional to its subjective validity. The chosen option is the one favored by the sampled cue. This naturally produces choice probabilities that reflect a mixture of top-cue and lower-cue usage across trials, naturally aligning with the ~0.50 choice shares observed in conflict trials without requiring extreme noise parameters."
}
```

## Usage

```json
{
  "prompt_token_count": 4782,
  "candidates_token_count": 327,
  "total_token_count": 6099
}
```
