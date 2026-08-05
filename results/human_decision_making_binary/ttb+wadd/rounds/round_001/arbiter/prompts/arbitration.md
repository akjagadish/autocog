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
Probabilistic Heuristic Selection: Subjects do not rigidly apply a single deterministic decision rule across all trials. Instead, they maintain a repertoire of strategies—specifically, a non-compensatory one-reason heuristic (Take The Best) and a compensatory strategy (Tallying/Equal-Weight). On each trial, a subject probabilistically selects which strategy to deploy based on a subject-specific mixing parameter (p_ttb). This trial-by-trial internal blending natively captures intermediate aggregate metric values while preserving low between-subject variance, as each individual exhibits a mixture of both choice patterns.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take The Best (Non-compensatory)
    ttb_winner = None
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 0
            break
        if b[j] > a[j]:
            ttb_winner = 1
            break
            
    if ttb_winner is None:
        p_ttb_arr = np.array([0.5, 0.5])
    else:
        p_ttb_arr = np.array([1.0, 0.0]) if ttb_winner == 0 else np.array([0.0, 1.0])
        
    # Strategy 2: Tallying (Compensatory)
    sum_a = np.sum(a)
    sum_b = np.sum(b)
    if sum_a > sum_b:
        p_tally_arr = np.array([1.0, 0.0])
    elif sum_b > sum_a:
        p_tally_arr = np.array([0.0, 1.0])
    else:
        p_tally_arr = np.array([0.5, 0.5])
        
    # Mixture
    p_ttb = float(parameters["p_ttb"])
    p_core = p_ttb * p_ttb_arr + (1.0 - p_ttb) * p_tally_arr
    
    # Late-stage lapse noise
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
    return int(np.argmax(probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=7):** [0.95, 0.94, 0.93, 0.92, 0.52, 0.51, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Probabilistic Heuristic Selection (PHS) model from the Weighted Additive (WADD) model, we exploit a 7-feature design with a steep drop in validities (four very high validities, three very low validities). Under PHS, subjects probabilistically mix Take The Best (TTB) and Tallying (Equal-Weight). We construct critical trials where Option A wins on the highest validity cue (favored by TTB) AND has a higher total number of positive features (favored by Tallying), meaning PHS strongly predicts choosing Option A regardless of the mixing parameter. However, Option B possesses several high-validity cues that collectively outweigh Option A's single highest-validity cue and its numerous low-validity cues. Thus, WADD strongly predicts choosing Option B. This provides a stark qualitative and quantitative divergence between the two models.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices that align with the Probabilistic Heuristic Selection (PHS) model on the critical trials (Trials 1 and 2). On these trials, PHS overwhelmingly predicts choosing the option with the highest validity cue (which also has the highest tally of positive features). WADD, on the other hand, integrates all cues and finds that the opponent option's multiple high-validity cues outweigh the single highest validity cue plus several low-validity ones. The metric will be close to 1.0 for PHS and close to 0.0 for WADD, providing a stark dissociation with very low between-subject variance.

Source:
def metric(data: pd.DataFrame) -> float:
    # Define the critical trials where PHS and WADD strongly diverge.
    # In Trial 1, Option A has the single highest validity cue and more total cues,
    # so PHS (TTB + Tallying) strongly predicts Option A (response 0).
    # Option B has multiple high-validity cues that outweigh A's cues in a weighted sum,
    # so WADD strongly predicts Option B (response 1).
    # Trial 2 is the exact symmetric swap.
    t1_A = (1, 0, 0, 0, 1, 1, 1)
    t2_A = (0, 1, 1, 1, 0, 0, 0)
    
    is_t1 = data['option_a_ratings'].apply(tuple) == t1_A
    is_t2 = data['option_a_ratings'].apply(tuple) == t2_A
    
    match_phs = 0
    match_phs += (data.loc[is_t1, 'response'] == 0).sum()
    match_phs += (data.loc[is_t2, 'response'] == 1).sum()
    
    total = is_t1.sum() + is_t2.sum()
    if total == 0:
        return 0.5
    return float(match_phs / total)


### RESULTS
- Predicted under pi_3 (simulated): 0.8050 (var=0.0118)
- Predicted under pi_2 (simulated): 0.2800 (var=0.1483)
- Observed on real data: 0.2317 (var=0.0651)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=8):** [0.99, 0.98, 0.97, 0.96, 0.55, 0.54, 0.53, 0.52]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  trial 6: A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) model from the Probabilistic Heuristic Selection (PHS) model, this experiment uses an 8-feature design with a steep drop in validities between the top four and bottom four features. PHS assumes subjects probabilistically mix Take The Best (TTB) and Tallying. In the critical trials (1-4), Option A has a '1' on the highest validity cue (winning TTB) and possesses a greater total number of positive features (winning Tallying), meaning PHS strongly predicts Option A regardless of the mixing parameter. However, Option B has '1's on the second, third, and fourth most valid cues, which collectively outweigh Option A's single highest-validity cue plus its numerous low-validity cues. Thus, WADD strongly predicts Option B. Trials 5-8 serve as baselines where both models largely agree.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
In the critical trials (Trials 1-4), Option A has the single most valid cue (cue 0) and a total of 4 positive features, whereas Option B has the next three most valid cues (cues 1, 2, 3) and only 3 positive features. The Probabilistic Heuristic Selection (PHS) model blends Take The Best (which strongly prefers A due to cue 0) and Tallying (which prefers A due to having more positive features). Thus, PHS almost exclusively chooses Option A on these trials, yielding a mean response near 0. Conversely, the Weighted Additive (WADD) model integrates all cues; the combined weight of the three highly valid cues for Option B will frequently outweigh Option A's single highly valid cue and three weakly valid cues, leading to a much higher proportion of Option B choices. The metric isolates these critical trials by filtering for A[0]==1 and B[1]==1, then calculates the proportion of times Option B is chosen.

Source:
def metric(data: pd.DataFrame) -> float:
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_1 = data['option_b_ratings'].apply(lambda x: x[1])
    critical_mask = (a_0 == 1) & (b_1 == 1)
    if critical_mask.sum() == 0:
        return 0.5
    return float(data.loc[critical_mask, 'response'].mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.0650 (var=0.0015)
- Predicted under pi_2 (simulated): 0.6300 (var=0.1806)
- Observed on real data: 0.8158 (var=0.0450)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        sum_a = sum(row['option_a_ratings'])
        sum_b = sum(row['option_b_ratings'])
        
        # In the compensatory trials designed for this experiment,
        # the option favored by TTB always has fewer total positive cues
        # than the option favored by WADD.
        if sum_a != sum_b:
            ttb_choice = 0 if sum_a < sum_b else 1
            ttb_match.append(1 if row['response'] == ttb_choice else 0)
            
    if not ttb_match:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.4175 (var=0.0635)
**Predicted under pi_3:** 0.4225 (var=0.0632)
**Predicted under pi_2:** 0.1000 (var=0.0400)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    ttb_matches = 0
    total = 0
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        response = row['response']
        
        ttb_choice = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        
        if ttb_choice is not None and response == ttb_choice:
            ttb_matches += 1
        total += 1
        
    return float(ttb_matches / total) if total > 0 else 0.0
```

**Observed (real) value:** 0.4317 (var=0.0770)
**Predicted under pi_3:** 0.4429 (var=0.0513)
**Predicted under pi_2:** 0.1933 (var=0.0482)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The data reveals a stark double dissociation between the two theories. In Experiments 1 and 2, where multiple cues have highly similar validities (e.g., 0.99, 0.98, 0.97), subjects consistently integrate these cues to override the single highest-validity cue, behaving in a compensatory manner that strongly favors WADD (pi_2) over the Probabilistic Heuristic Selection (pi_3) model. However, in Experiments 3 and 4, the observed choice proportions perfectly match the predictions of pi_3 (PHS), while WADD drastically underpredicts the rate of heuristic-like choices. Neither theory is task-invariant: pi_3 fails to allow for weighted compensatory integration of close-validity cues, and pi_2 fails to capture the high baseline rate of non-compensatory or tallying-like choices. A new theory is needed to reconcile these findings.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a brand-new theory that unifies the compensatory integration of WADD with the heuristic behavior of TTB/Tallying. One promising direction is a 'Non-linear Subjective Weighting' model where subjective cue weights are a power function of the provided validities (with an individual-difference parameter controlling the exponent). A high exponent leads to non-compensatory (TTB-like) behavior, while an exponent near 1 yields WADD-like behavior, and an exponent near 0 yields Tallying. Alternatively, propose an 'Adaptive Strategy Selection' theory where subjects probabilistically mix WADD and TTB rather than Tallying and TTB."
}
```

## Usage

```json
{
  "prompt_token_count": 5747,
  "candidates_token_count": 369,
  "total_token_count": 7342
}
```
