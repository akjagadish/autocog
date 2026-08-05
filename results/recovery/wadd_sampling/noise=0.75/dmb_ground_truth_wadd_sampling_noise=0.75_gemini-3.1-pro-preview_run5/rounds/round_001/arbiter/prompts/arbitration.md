# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_1" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_1" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


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


## THEORY 2 — pi_3
Weighted Additive (WADD) Theory with Subjective Validity Scaling: People compare options by computing a comprehensive value for each option, integrating all available cues by weighting each feature. However, instead of using raw objective validities, individuals apply a non-linear subjective scaling to these validities. This scaling allows a single highly valid cue to be disproportionately weighted to balance out multiple less valid cues, gracefully explaining why individuals exhibit near-indifference (choice proportions around 0.5) when faced with adversarial choices where one option excels on the best cue but the other excels on numerous lesser cues. Response noise is captured via a softmax over the options' weighted sums, along with an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
    
    # Apply non-linear subjective scaling to validities
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Compute weighted additive values for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=5):** [0.96, 0.84, 0.72, 0.6, 0.5]

**Trial pairs (n=14):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 10: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 11: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 12: A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 13: A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 0]
  trial 14: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Take The Best (TTB) from Weighted Additive (WADD) with subjective validity scaling, we vary the number of lower-validity cues that contradict the highest-validity discriminating cue. For TTB, choices are determined exclusively by the first discriminating cue, meaning choice probabilities will be identical across trials that share the same top discriminator, regardless of the number of opposing lower-validity cues. In contrast, WADD integrates all cues, so it predicts a parametric decrease in the choice probability for the option favored by the best cue as the number of opposing lower-validity cues increases.

**Computed schedule:** 14 unique pairs × 6 reps = 84 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the probability of choosing the option favored by the single best discriminating cue (the TTB winner) when there is only 1 opposing cue versus when there are 3 or more opposing cues. According to Take The Best, choices are based exclusively on the top discriminating cue, so the number of opposing lower-validity cues should not affect the choice probability, yielding a difference near zero. According to Weighted Additive (WADD) theory, choices integrate all cues, so a higher number of opposing cues parametrically decreases the likelihood of choosing the option favored by the best cue, yielding a strictly positive difference.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    ttb_chose_winner = []
    opposing_cues = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        winner_ttb = None
        for j in range(len(a)):
            if a[j] > b[j]:
                winner_ttb = 0
                break
            elif b[j] > a[j]:
                winner_ttb = 1
                break
                
        if winner_ttb is None:
            continue
            
        opp = 0
        for j in range(len(a)):
            if winner_ttb == 0 and b[j] > a[j]:
                opp += 1
            elif winner_ttb == 1 and a[j] > b[j]:
                opp += 1
                
        opposing_cues.append(opp)
        ttb_chose_winner.append(1 if row['response'] == winner_ttb else 0)
        
    df = pd.DataFrame({'opp': opposing_cues, 'chose_ttb': ttb_chose_winner})
    
    p_1 = df[df['opp'] == 1]['chose_ttb'].mean()
    p_3 = df[df['opp'] >= 3]['chose_ttb'].mean()
    
    if pd.isna(p_1) or pd.isna(p_3):
        return 0.0
        
    return float(p_1 - p_3)

### RESULTS
- Predicted under pi_1 (simulated): -0.0197 (var=0.0064)
- Predicted under pi_3 (simulated): 0.2686 (var=0.0732)
- Observed on real data: 0.0456 (var=0.0198)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=14):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 10: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 11: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 13: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 14: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** To quantitatively dissociate WADD from TTB, we hold the top discriminating cue constant while systematically varying the *validity* (rather than just the number) of the opposing cues. TTB bases its decision entirely on the first discriminating cue and ignores the magnitude of any lower-validity cues, predicting identical choice probabilities across these trials. Tallying also fails to distinguish between these trials since the number of opposing cues remains constant. WADD, however, integrates the subjective weights of all cues, predicting a parametric shift in choice probability depending on whether the opposing evidence comes from high-validity or low-validity secondary cues.

**Computed schedule:** 14 unique pairs × 6 reps = 84 trials per subject.



### METRIC
Rationale:
To maximize the divergence between WADD and TTB, this metric contrasts the most extreme trial types available. In 'weak opponent' trials, the TTB-favored option faces an alternative with only the lowest-validity cue. Both WADD and TTB strongly prefer the TTB option here. In 'strong opponent' trials, the TTB option faces an alternative with BOTH the second and third best cues. TTB completely ignores these secondary cues and consistently picks the TTB option. However, WADD integrates the evidence; because the sum of the second and third cue validities (0.85 + 0.75 = 1.6) heavily outweighs the top cue (0.95) under moderate scaling, WADD reverses its preference and strongly favors the opponent. Thus, WADD predicts a massive drop in the TTB-match rate between these two conditions, whereas TTB predicts zero difference.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify which option is favored by Take The Best (the one with 1 on the top cue)
    ttb_favored_is_A = data['option_a_ratings'].apply(lambda x: x[0] == 1)
    ttb_match = ((data['response'] == 0) == ttb_favored_is_A).astype(float)
    
    # Extract the features of the opposing (non-TTB-favored) option
    def get_opponent(row):
        if row['option_a_ratings'][0] == 1:
            return tuple(row['option_b_ratings'])
        else:
            return tuple(row['option_a_ratings'])
            
    opp = data.apply(get_opponent, axis=1)
    
    # Trials where the opposing option is very weak (only has the 5th best cue)
    weak_opp = opp == (0, 0, 0, 0, 1)
    # Trials where the opposing option is very strong (has both the 2nd and 3rd best cues)
    strong_opp = opp == (0, 1, 1, 0, 0)
    
    val_weak = ttb_match[weak_opp].mean() if weak_opp.any() else 0.5
    val_strong = ttb_match[strong_opp].mean() if strong_opp.any() else 0.5
    
    # Return the difference in choice probability for the TTB-favored option
    return float(val_weak - val_strong)

### RESULTS
- Predicted under pi_1 (simulated): 0.0050 (var=0.0196)
- Predicted under pi_3 (simulated): 0.2983 (var=0.0789)
- Observed on real data: 0.0600 (var=0.0436)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(np.mean(data['response'] == 0))
```

**Observed (real) value:** 0.4554 (var=0.0025)
**Predicted under pi_1:** 0.8592 (var=0.0090)
**Predicted under pi_3:** 0.4731 (var=0.0503)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    tally_pred = (sum_b > sum_a).astype(int)
    return float((data['response'] == tally_pred).mean())
```

**Observed (real) value:** 0.5387 (var=0.0030)
**Predicted under pi_1:** 0.1506 (var=0.0094)
**Predicted under pi_3:** 0.5256 (var=0.0576)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the four experiments, we see a striking dissociation. In Experiments 1 and 2, the observed data show only a very small effect of opposing lower-validity cues (metrics ~0.04 to 0.06), which is much closer to the TTB (pi_1) prediction of ~0 than the WADD (pi_3) prediction of ~0.27-0.29. However, in Experiments 3 and 4, TTB fails catastrophically: it predicts massive preferences for the option with the single best cue, whereas observed data show near-indifference or a slight preference for the option with many lower-validity cues, which WADD captures perfectly. Neither pure TTB nor pure WADD can simultaneously explain the robustness to opposing cues in Exps 1/2 and the compensatory behavior in Exps 3/4. This suggests a mixed population or a strategy-selection mechanism.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Strategy Mixture' or 'Dual Process' theory. The new theory should posit that decision-makers are heterogenous (or use different strategies on different trials): a proportion of choices are made using a strict non-compensatory heuristic like Take-The-Best, while the remainder are made using a compensatory integration strategy (like Tallying or WADD). This mixture would correctly predict a dampened but non-zero effect of opposing cues in Experiments 1 and 2, while successfully capturing the compensatory shifts seen in Experiments 3 and 4 that a pure TTB model completely misses."
}
```

## Usage

```json
{
  "prompt_token_count": 6053,
  "candidates_token_count": 354,
  "total_token_count": 7534
}
```
