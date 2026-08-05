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
Take-The-Best (TTB) heuristic: People do not integrate all available features when choosing between options. Instead, they employ a non-compensatory, lexicographic strategy. They inspect cues one by one in descending order of their validity. The search stops at the first cue that discriminates between the two options (i.e., one option has a higher value than the other on that cue), and the option with the higher value on this discriminating cue is chosen. If no cue discriminates, they guess.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort features by validity in descending order
    order = np.argsort(validities)[::-1]
    
    a, b = stim[0], stim[1]
    scores = np.array([0.0, 0.0])
    
    # Lexicographic search: stop at the first discriminating cue
    for idx in order:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the scores (which are either [1, 0], [0, 1], or [0, 0] if tied)
    z = beta * scores
    z = z - np.max(z)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate Take-The-Best (TTB) from Weighted Additive (WADD), we exploit TTB's strict non-compensatory lexicographic stopping rule. By configuring the validities such that the most valid cue is outweighed by the sum of multiple lower-validity cues, we create critical trials where the two models make opposite predictions. For instance, if Option A wins on the highest validity cue but Option B wins on all remaining cues, TTB will deterministically choose A (ignoring the rest of the features), whereas WADD will integrate all features and strongly prefer B. The trials include a mix of these compensatory vs. non-compensatory conflicts at various cue validities, as well as control trials where both models agree.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
In all 6 trial types, the Take-The-Best (TTB) model will deterministically select Option A because Option A always has the advantage on the highest-validity cue that discriminates the two options. In contrast, the Weighted Additive (WADD) model integrates all cues, and in most of these trials, Option B has a higher sum of weighted validities (compensatory advantage from multiple lower-validity cues). Therefore, the overall proportion of choosing Option A will be near 1.0 for TTB and much lower for WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    return float(np.mean(data['response'] == 0))

### RESULTS
- Predicted under pi_3 (simulated): 0.8408 (var=0.0088)
- Predicted under pi_2 (simulated): 0.3550 (var=0.0231)
- Observed on real data: 0.5121 (var=0.0035)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 4: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 6: A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 9: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 11: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 12: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) rule from the Take-The-Best (TTB) heuristic, we exploit TTB's strict non-compensatory lexicographic stopping rule. We configure the validities such that the most valid cues can be outweighed by the sum of several lower-validity cues. In critical trials, one option wins on the highest-validity cue (thus TTB deterministically chooses it, ignoring all other features), whereas the other option wins on multiple remaining cues (causing WADD to integrate the features and prefer the latter option). By including trials where the conflict happens at the first cue, the second cue, and the third cue, we can robustly distinguish whether subjects use a lexicographic search or a compensatory weighted integration.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
We measure the proportion of choices that align with the Take-The-Best (TTB) heuristic. Because the experiment consists mostly of 'compensatory' trials where WADD and TTB predict opposite responses (e.g., one option wins on the single most valid cue, while the other wins on several less valid cues that together outweigh the first), this metric will be high for data generated by TTB and low for data generated by WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    matches = []
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        pred = -1
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        if pred != -1:
            matches.append(1 if r == pred else 0)
    return float(np.mean(matches)) if matches else 0.0

### RESULTS
- Predicted under pi_3 (simulated): 0.8435 (var=0.0120)
- Predicted under pi_2 (simulated): 0.4254 (var=0.0137)
- Observed on real data: 0.5188 (var=0.0033)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  A=[0, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_f0 = np.array([x[0] for x in data['option_a_ratings']])
    a_f1 = np.array([x[1] for x in data['option_a_ratings']])
    b_f0 = np.array([x[0] for x in data['option_b_ratings']])
    b_f1 = np.array([x[1] for x in data['option_b_ratings']])
    
    a_sum2 = a_f0 + a_f1
    b_sum2 = b_f0 + b_f1
    
    # Identify critical trials (trials 1 and 2) where one option has the two highest 
    # validity features (sum=2) and the other has none of them (sum=0) but wins on the rest.
    mask = ((a_sum2 == 0) & (b_sum2 == 2)) | ((a_sum2 == 2) & (b_sum2 == 0))
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    a_sum2_rel = a_sum2[mask]
    
    # Tallying prefers the option with more features (which here means the one with 0 on the first two features)
    tally_chose_a = (a_sum2_rel == 0) & (responses == 0)
    tally_chose_b = (a_sum2_rel == 2) & (responses == 1)
    
    return float(np.mean(tally_chose_a | tally_chose_b))

```

**Observed (real) value:** 0.4867 (var=0.0082)
**Predicted under pi_3:** 0.1608 (var=0.0119)
**Predicted under pi_2:** 0.5358 (var=0.1008)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    wadd_choices = 0
    total = 0
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        if a[0] == 1 and a[1] == 1 and b[0] == 0 and b[1] == 0:
            if resp == 0:
                wadd_choices += 1
            total += 1
        elif a[0] == 0 and a[1] == 0 and b[0] == 1 and b[1] == 1:
            if resp == 1:
                wadd_choices += 1
            total += 1
            
    if total == 0:
        return 0.5
    return float(wadd_choices / total)

```

**Observed (real) value:** 0.5089 (var=0.0125)
**Predicted under pi_3:** 0.8411 (var=0.0167)
**Predicted under pi_2:** 0.5700 (var=0.1165)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metric values consistently hover around 0.50 with very low variance (e.g., ~0.51 in Exp 1 and 2, ~0.49 in Exp 3, ~0.51 in Exp 4). These experiments were specifically designed with critical trials where a non-compensatory strategy (Take-The-Best) and a compensatory strategy (Weighted Additive or Tallying) make opposite predictions. Theory 1 (TTB) consistently predicts extreme values (~0.84 or ~0.16) because it expects a deterministic lexicographic choice. Theory 2 (WADD) predicts values closer to the middle but with extremely high between-subject variance (e.g., ~0.11), which does not match the very low observed variance (~0.003). The fact that the population averages ~0.50 on these conflict trials strongly suggests a stable mixture of strategies across the population (e.g., half the subjects/trials use TTB, half use a compensatory rule), or that the conflict between cue validity and cue quantity causes subjects to guess. Neither pure TTB nor pure WADD captures this systematic 50/50 split.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Mixture' theory. Instead of assuming all subjects use a single heuristic (like pure TTB or pure WADD), this theory should posit that individuals draw from a repertoire of strategies. Specifically, on any given trial, a subject might use a lexicographic rule (TTB) with probability `p_ttb`, and a compensatory rule (like Tallying or WADD) with probability `1 - p_ttb`. This mixture will naturally predict the ~0.50 observed rates on conflict trials while maintaining low variance if the mixture proportion is relatively stable across the population."
}
```

## Usage

```json
{
  "prompt_token_count": 5447,
  "candidates_token_count": 417,
  "total_token_count": 6572
}
```
