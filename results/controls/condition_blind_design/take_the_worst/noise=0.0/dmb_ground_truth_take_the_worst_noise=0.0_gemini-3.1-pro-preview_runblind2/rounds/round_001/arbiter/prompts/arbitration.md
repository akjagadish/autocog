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
People make binary choices by evaluating all available features and weighting them proportionally to their validities. This Weighted Additive (WADD) model is a compensatory strategy: a strong disadvantage on one highly valid cue can be compensated for by advantages on several less valid cues. This contrasts with non-compensatory heuristics like Take The Best, which ignores lower-validity cues entirely, and equal-weighting heuristics like Tallying, which ignore the relative importance of different cues.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # WADD: Each option's score is the sum of its cue values weighted by cue validity.
    score_a = np.sum(val * stim[0])
    score_b = np.sum(val * stim[1])
    
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_2
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


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.77, 0.8, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[1, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 2: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 3: A=[1, 1, 0, 1]  B=[0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1]  B=[0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1]  B=[0, 0, 0, 1]
  trial 7: A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 8: A=[1, 1, 0, 0]  B=[1, 1, 1, 0]
  trial 9: A=[1, 0, 1, 1]  B=[0, 1, 1, 1]
  trial 10: A=[0, 0, 0, 0]  B=[0, 0, 1, 1]
  trial 11: A=[0, 0, 0, 1]  B=[1, 0, 1, 1]
  trial 12: A=[0, 1, 1, 0]  B=[1, 1, 1, 0]
  trial 13: A=[0, 0, 0, 1]  B=[1, 0, 0, 0]
  trial 14: A=[1, 0, 1, 0]  B=[0, 1, 1, 0]
  trial 15: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 16: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates trials where the Tallying heuristic faces a tie (both options have the same number of winning features) and is forced to guess, leading to a 50% chance of choosing either option. On these exact same trials, the Weighted Additive (WADD) model often has a strict preference due to the differing validities of the features. By calculating the proportion of choices that align with the WADD prediction on Tally-tie trials, we can cleanly separate the two theories. Data simulated under Tallying will yield a metric close to 0.5, whereas data simulated under WADD will yield a metric significantly higher than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.77, 0.8, 0.55])
    match_count = 0
    tie_count = 0
    
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        # Check if it's a tie under Tallying
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins == b_wins:
            tie_count += 1
            
            # WADD predictions
            wadd_a = np.sum(validities * a)
            wadd_b = np.sum(validities * b)
            
            if wadd_a > wadd_b:
                wadd_pred = 0
            elif wadd_b > wadd_a:
                wadd_pred = 1
            else:
                continue
                
            if row['response'] == wadd_pred:
                match_count += 1
                
    if tie_count == 0:
        return 0.5
        
    return match_count / tie_count


### RESULTS
- Predicted under pi_3 (simulated): 0.7673 (var=0.0170)
- Predicted under pi_2 (simulated): 0.5080 (var=0.0084)
- Observed on real data: 0.1240 (var=0.0095)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=4):** [0.95, 0.84, 0.64, 0.55]

**Trial pairs (n=16):**
  trial 1: A=[0, 0, 1, 1]  B=[1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[1, 0, 1, 1]
  trial 4: A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  trial 5: A=[1, 0, 1, 1]  B=[0, 0, 1, 1]
  trial 6: A=[1, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 7: A=[0, 1, 0, 1]  B=[1, 1, 0, 0]
  trial 8: A=[1, 1, 0, 0]  B=[0, 1, 1, 0]
  trial 9: A=[1, 1, 0, 0]  B=[1, 0, 0, 1]
  trial 10: A=[0, 0, 0, 1]  B=[0, 1, 0, 0]
  trial 11: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 12: A=[1, 0, 1, 0]  B=[0, 0, 1, 0]
  trial 13: A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  trial 14: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  trial 15: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]
  trial 16: A=[1, 1, 1, 1]  B=[1, 1, 0, 1]

**Rationale:** (no rationale)

**Computed schedule:** 16 unique pairs × 6 reps = 96 trials per subject.



### METRIC
Rationale:
In the experimental design, there are several trials where the two options win on an equal number of features. The Tallying heuristic considers these to be ties and predicts that subjects will guess randomly (50% probability). In contrast, the Weighted Additive (WADD) model heavily favors the option with the higher cue validities on these exact trials. By measuring the proportion of times subjects choose the WADD-preferred option on Tallying-tie trials, we can cleanly separate the two models: Tallying predicts a value of 0.5, while WADD predicts a value significantly greater than 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.84, 0.64, 0.55])
    
    match_wadd = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        # Focus on trials where Tallying sees a tie
        if a_wins == b_wins:
            wadd_a = np.sum(val * a)
            wadd_b = np.sum(val * b)
            
            if wadd_a > wadd_b:
                pref = 0
            elif wadd_b > wadd_a:
                pref = 1
            else:
                continue
                
            match_wadd.append(1 if row['response'] == pref else 0)
            
    if len(match_wadd) == 0:
        return 0.5
        
    return float(np.mean(match_wadd))

### RESULTS
- Predicted under pi_3 (simulated): 0.7539 (var=0.0177)
- Predicted under pi_2 (simulated): 0.5056 (var=0.0059)
- Observed on real data: 0.1589 (var=0.0095)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 1, 0, 1]  B=[1, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[1, 1, 1, 0]
  A=[1, 0, 0, 1]  B=[1, 1, 0, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[1, 0, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 0, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[1, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        resp = row['response']
        
        # Trial 6: TTB picks B (cue 0), Tallying picks A (A wins 2 cues to 1)
        if a == (0, 1, 1, 1) and b == (1, 1, 0, 0):
            ttb_match.append(1 if resp == 1 else 0)
        # Trials 8 & 16: TTB picks A (cue 0), Tallying picks B (B wins 2 cues to 1)
        elif a == (1, 1, 0, 0) and b == (0, 1, 1, 1):
            ttb_match.append(1 if resp == 0 else 0)
            
    if not ttb_match:
        return 0.5
    return float(np.mean(ttb_match))
```

**Observed (real) value:** 0.1733 (var=0.0250)
**Predicted under pi_3:** 0.1622 (var=0.0175)
**Predicted under pi_2:** 0.1689 (var=0.0174)

### Experiment 4
**Design**
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 0, 0, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[1, 1, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 1, 0]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 1]
  A=[1, 1, 1, 0]  B=[0, 0, 1, 0]
  A=[1, 0, 1, 0]  B=[0, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 1, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 0]  B=[0, 1, 0, 1]
  A=[1, 1, 1, 1]  B=[1, 0, 0, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Tallying tallies strict wins across all features
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    tally_prefers_a = a_wins > b_wins
    
    # TTB checks the most valid cue first (cue 0, validity 0.95)
    ttb_prefers_b = b_mat[:, 0] > a_mat[:, 0]
    
    # Identify conflict trials where Tallying prefers A but TTB prefers B
    conflict_mask = tally_prefers_a & ttb_prefers_b
    
    # Return the proportion of times B was chosen on these conflict trials
    # Tallying will yield ~0.0, TTB will yield ~1.0
    if np.any(conflict_mask):
        return float(data.loc[conflict_mask, 'response'].mean())
    return 0.5

```

**Observed (real) value:** 0.1267 (var=0.0206)
**Predicted under pi_3:** 0.1867 (var=0.0418)
**Predicted under pi_2:** 0.1600 (var=0.0255)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "The data across the experiments reveals a clear, consistent pattern that neither the Weighted Additive (WADD) model nor the pure Tallying heuristic can explain. In Experiments 3 and 4, when Tallying and Take-The-Best (TTB) conflict, subjects overwhelmingly choose the option favored by Tallying, contradicting TTB. However, in Experiments 1 and 2, on trials where Tallying results in a tie (both options win on an equal number of features), subjects do not simply guess randomly (as pure Tallying predicts, expected value 0.5). Instead, they systematically choose the option with the lower WADD score (observed values ~0.12-0.15). This indicates that when the number of winning features is tied, subjects break the tie by looking at the single most valid cue, which often opposes the WADD sum of the remaining lower-validity cues. Subjects are employing a two-stage heuristic: Majority Rule (Tallying) first, followed by Take-The-Best to break ties.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Majority Rule with Lexicographic Tie-Breaking' theory. The theory should state that subjects first compare options by tallying the number of winning features for each (equal weighting). If one option wins on more features, it is chosen. If there is a tie in the tally, subjects break the tie non-compensatorily by choosing the option that wins on the single most valid cue. This explains why they follow Tallying in Exp 3/4 but strongly oppose WADD on Tallying-tie trials in Exp 1/2."
}
```

## Usage

```json
{
  "prompt_token_count": 6038,
  "candidates_token_count": 374,
  "total_token_count": 8007
}
```
