# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_2") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_2" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Strategy Mixture Theory: Instead of relying on a single heuristic, decision-makers draw from a repertoire of strategies on a trial-by-trial basis. Specifically, individuals mix between a non-compensatory lexicographic rule (Take-The-Best) and a compensatory rule (Tallying). On any given trial, a subject employs TTB with probability `p_ttb` and Tallying with probability `1 - p_ttb`. This intra-individual strategy variation naturally accounts for the aggregate ~0.50 choice proportions observed in conflict trials where the two heuristics prescribe different options, while a relatively stable mixture proportion across the population explains the low between-subject variance.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Strategy Mixture expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    # Strategy 1: Take-The-Best (TTB)
    order = np.argsort(validities)[::-1]
    score_ttb = np.array([0.0, 0.0])
    for idx in order:
        if a[idx] > b[idx]:
            score_ttb[0] = 1.0
            break
        elif b[idx] > a[idx]:
            score_ttb[1] = 1.0
            break
            
    # Strategy 2: Tallying (Compensatory)
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    score_tally = np.array([a_wins, b_wins])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb = float(parameters["p_ttb"])
    
    # Softmax for TTB
    z_ttb = beta * score_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    prob_ttb = e_ttb / np.sum(e_ttb)
    
    # Softmax for Tallying
    z_tally = beta * score_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    prob_tally = e_tally / np.sum(e_tally)
    
    # Mix the two strategies
    p_core = p_ttb * prob_ttb + (1.0 - p_ttb) * prob_tally
    
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


## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the Strategy Mixture Theory (which posits intra-individual trial-by-trial mixing of Take-The-Best and Tallying) from the Weighted Additive (WADD) rule. The validities are structured with two highly valid cues and three less valid cues. We construct critical trials where TTB and Tallying point in opposite directions (e.g., Option A wins on the highest validity cue, but Option B wins on a greater number of less valid cues). Because Strategy Mixture Theory assumes individuals mix TTB and Tallying (with p_ttb ~ 0.5), it predicts a choice proportion near 0.5 for these conflict trials. In contrast, WADD integrates all cardinal validities and will strongly favor one option depending on the exact sum of weights, driving choice probabilities toward 1 or 0. By varying which option WADD favors while keeping the TTB/Tallying conflict constant, we can perfectly dissociate the two models.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
Strategy Mixture Theory posits that individuals mix between TTB and Tallying on a trial-by-trial basis. On conflict trials where the two heuristics disagree, an SMT agent will stochasticly alternate between the two options across repetitions of the exact same trial, yielding a choice proportion near 0.5 (low extremity). In contrast, WADD computes a deterministic weighted sum based on subjective validities. For any specific trial, a WADD agent will consistently favor one option over the other across all repetitions (modulo some response noise), yielding choice proportions closer to 0 or 1 (high extremity). By measuring the absolute deviation of the intra-individual choice proportion from 0.5 on identical conflict trials, we can cleanly separate the stochastic strategy mixing of SMT from the deterministic weighting of WADD.

Source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    # Create a hashable trial identifier
    data['trial_id'] = data.apply(
        lambda row: (tuple(row['option_a_ratings']), tuple(row['option_b_ratings'])), 
        axis=1
    )
    
    # Identify conflict trials where TTB and Tallying prescribe different options.
    # TTB relies on the first cue (index 0). Tallying relies on the sum of cues.
    def is_conflict(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        ttb_a = a[0] > b[0]
        ttb_b = b[0] > a[0]
        tally_a = sum(a) > sum(b)
        tally_b = sum(b) > sum(a)
        
        if ttb_a and tally_b:
            return True
        if ttb_b and tally_a:
            return True
        return False

    data['is_conflict'] = data.apply(is_conflict, axis=1)
    conflict_data = data[data['is_conflict']]
    
    if len(conflict_data) == 0:
        return 0.0
        
    # For each subject and each unique conflict trial, compute the choice proportion.
    # response == 0 means A, response == 1 means B. 
    # The mean of response is the proportion of B choices (p_B).
    # The absolute difference from 0.5 measures how deterministic the subject's choice is.
    grouped = conflict_data.groupby(['subject_id', 'trial_id'])['response'].mean().reset_index()
    grouped['extremity'] = (grouped['response'] - 0.5).abs()
    
    # Average the extremity of preferences per subject across all conflict trials,
    # then return the grand mean.
    subj_extremity = grouped.groupby('subject_id')['extremity'].mean()
    return float(subj_extremity.mean())


### RESULTS
- Predicted under pi_4 (simulated): 0.1108 (var=0.0011)
- Predicted under pi_2 (simulated): 0.2882 (var=0.0098)
- Observed on real data: 0.1010 (var=0.0012)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.55, 0.52, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** This design specifically targets the Strategy Mixture model, which posits that subjects randomly alternate between Take-The-Best (TTB) and Tallying. By pitting TTB against Tallying across a set of trials, the Strategy Mixture model predicts choice proportions near 0.50 (assuming a typical mixing probability around 0.50). However, the Weighted Additive (WADD) model integrates all cues according to their validities. By carefully selecting the validities and feature configurations, we can force WADD to strongly prefer Option A in some of these conflict trials and Option B in others, whereas Strategy Mixture remains stuck at predicting ~0.50 across all of them.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric computes the difference in the probability of choosing Option A between Trial 2 and Trial 4. In both of these trials, the Strategy Mixture model pits Take-The-Best (which chooses A) against Tallying (which chooses B), resulting in a predicted choice proportion for A equal to the mixing parameter `p_ttb` (around 0.50) in both cases, making the expected difference 0. However, the Weighted Additive (WADD) model strongly differentiates the two trials: Option A in Trial 2 has an additional high-validity feature compared to Option A in Trial 4, while Option B remains identical. Consequently, WADD predicts a substantially higher choice probability for A in Trial 2 than in Trial 4. This difference will be positive and large under WADD, but zero under Strategy Mixture.

Source:
def metric(data: pd.DataFrame) -> float:
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 2: A=[1, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t2_mask = a_tuples == (1, 1, 0, 0, 0)
    # Trial 4: A=[0, 1, 0, 0, 0], B=[0, 0, 1, 1, 1]
    t4_mask = a_tuples == (0, 1, 0, 0, 0)
    
    if not t2_mask.any() or not t4_mask.any():
        return 0.0
        
    p_a_t2 = (data.loc[t2_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    return float(p_a_t2 - p_a_t4)

### RESULTS
- Predicted under pi_4 (simulated): 0.0032 (var=0.0287)
- Predicted under pi_2 (simulated): 0.2305 (var=0.1023)
- Observed on real data: -0.0084 (var=0.0161)

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
**Predicted under pi_4:** 0.4908 (var=0.0117)
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
**Predicted under pi_4:** 0.5067 (var=0.0155)
**Predicted under pi_2:** 0.5700 (var=0.1165)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    return float(np.mean(data['response'] == 0))
```

**Observed (real) value:** 0.5121 (var=0.0035)
**Predicted under pi_4:** 0.5231 (var=0.0036)
**Predicted under pi_2:** 0.3550 (var=0.0231)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
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
```

**Observed (real) value:** 0.5188 (var=0.0033)
**Predicted under pi_4:** 0.5460 (var=0.0047)
**Predicted under pi_2:** 0.4254 (var=0.0137)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, Strategy Mixture Theory (THEORY 1, pi_4) consistently and accurately predicts the observed data, while the Weighted Additive model (THEORY 2, pi_2) is consistently off. In Experiment 1, the low choice extremity (0.10) closely matches the stochastic mixing of heuristics rather than deterministic WADD. In Experiment 2, adding an extra high-validity cue to Option A did not increase its choice probability (observed difference ~ -0.008), which directly contradicts WADD but is perfectly predicted by Strategy Mixture Theory. Experiments 3-6 further confirm that subjects do not integrate all cues in a weighted compensatory manner, but rather show choice proportions near 0.5 on conflict trials. WADD fails to capture this behavioral pattern.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Replace the Weighted Additive (WADD) theory with a 'Sequential Evidence Accumulation' theory (e.g., a discrete Multi-Attribute Decision Field Theory or an Evidence Threshold model). Instead of positing an ad-hoc trial-by-trial coin flip between two distinct heuristic algorithms (TTB and Tallying), this new theory should propose a unified mechanism: decision-makers sample cues sequentially (e.g., in order of validity) and accumulate evidence. If the accumulated evidence reaches a certain internal threshold, a choice is made; if all cues are exhausted without reaching the threshold, they guess. By varying the threshold, this single mechanism can naturally produce both TTB-like behavior (low threshold) and Tallying-like behavior (high threshold), offering a more mechanistic and unified explanation for the observed stochasticity and conflict-trial choice proportions."
}
```

## Usage

```json
{
  "prompt_token_count": 7159,
  "candidates_token_count": 377,
  "total_token_count": 8475
}
```
