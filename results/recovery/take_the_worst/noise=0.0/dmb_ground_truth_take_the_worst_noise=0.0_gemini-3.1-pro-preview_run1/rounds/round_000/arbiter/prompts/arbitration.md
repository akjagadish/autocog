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


## EXPERIMENT 1 (proposed by pi_1)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  trial 6: A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 7: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Rationale:** This design systematically pits Take The Best (TTB) against Tallying by varying which option wins on the highest-validity discriminating cue versus which option wins on the sheer number of cues. By assigning strictly descending validities to the 4 features, TTB will always select the option that wins on the earliest discriminating feature. In contrast, Tallying simply counts the number of winning features, completely ignoring their validities. The selected trials include strong dissociations where TTB chooses one option (because it wins on the single most valid cue) while Tallying confidently chooses the other (because it wins on a larger number of lower-validity cues), as well as cases where TTB strongly prefers one option but Tallying predicts a tie.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric isolates the trials where Take The Best (TTB) and Tallying make strictly opposing deterministic predictions. It calculates the proportion of these diagnostic trials on which the subject's choice aligns with TTB. Data generated by TTB will yield values near 1, whereas data generated by Tallying will yield values near 0, maximizing the discriminability between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.array(data['option_a_ratings'].tolist())
    b_mat = np.array(data['option_b_ratings'].tolist())
    resp = data['response'].values
    
    ttb_choices = np.zeros(len(data))
    mask = np.zeros(len(data), dtype=bool)
    
    for i in range(len(data)):
        a = a_mat[i]
        b = b_mat[i]
        
        # TTB choice (first discriminating cue)
        ttb_c = -1
        for j in range(len(a)):
            if a[j] > b[j]:
                ttb_c = 0
                break
            elif b[j] > a[j]:
                ttb_c = 1
                break
                
        # Tallying choice (most feature wins)
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        tally_c = -1
        if a_wins > b_wins:
            tally_c = 0
        elif b_wins > a_wins:
            tally_c = 1
            
        # Only consider trials where TTB and Tallying make strictly opposing predictions
        if ttb_c != -1 and tally_c != -1 and ttb_c != tally_c:
            mask[i] = True
            ttb_choices[i] = ttb_c
            
    if not np.any(mask):
        return 0.5
        
    return float(np.mean(resp[mask] == ttb_choices[mask]))

### RESULTS
- Predicted under pi_1 (simulated): 0.8742 (var=0.0109)
- Predicted under pi_2 (simulated): 0.1297 (var=0.0093)
- Observed on real data: 0.1383 (var=0.0087)

## EXPERIMENT 2 (proposed by pi_2)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Rationale:** This design systematically pits Tallying against Take The Best (TTB) across 5 features with descending validities. Tallying ignores validities and simply counts the number of features favoring each option, whereas TTB relies exclusively on the highest-validity discriminating feature. The trial pairs are constructed to create sharp dissociations: in many trials, one option wins on the single most valid cue (thus chosen by TTB) while the other option wins on a sheer majority of lower-validity cues (thus chosen by Tallying). We also include trials where Tallying predicts a tie (equal number of winning features) while TTB makes a deterministic prediction based on the most valid cue.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The experimental design pits Tallying against Take The Best by constructing trials where the two heuristics make exactly opposite deterministic predictions. This metric computes the proportion of choices that match the Tallying prediction on all trials where Tallying has a strict preference. Under Tallying, this proportion will be significantly greater than 0.5 (approaching 1 depending on noise). Under Take The Best, because it predicts the opposite on these specific trials, the proportion will be significantly less than 0.5 (approaching 0). This creates a massive, reliable gap between the theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    matches = []
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        
        a_wins = np.sum(a > b)
        b_wins = np.sum(b > a)
        
        if a_wins > b_wins:
            tally_pred = 0
        elif b_wins > a_wins:
            tally_pred = 1
        else:
            continue
            
        matches.append(row['response'] == tally_pred)
        
    if not matches:
        return 0.5
    return float(np.mean(matches))

### RESULTS
- Predicted under pi_1 (simulated): 0.1536 (var=0.0070)
- Predicted under pi_2 (simulated): 0.8479 (var=0.0103)
- Observed on real data: 0.8286 (var=0.0105)

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
  "interpretation": "Across both experiments, Theory 2 (Tallying) perfectly captures the observed behavior, while Theory 1 (Take The Best) completely fails. In Experiment 1, subjects chose the TTB prediction only ~14% of the time on opposing trials, closely matching the 13% predicted by Tallying. In Experiment 2, subjects chose the Tallying prediction ~83% of the time, aligning with the 85% predicted by Tallying and contradicting the 15% predicted by TTB. Since Tallying is doing an excellent job and TTB is thoroughly falsified in this domain, TTB should be replaced with a stronger compensatory competitor.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a Weighted Additive (WADD) theory. Unlike Tallying (which weights all features equally) and TTB (which only uses the single best discriminating feature), WADD computes a weighted sum of all features, where the weights are derived from the feature validities (e.g., log-odds or directly proportional to validity). This provides a natural compensatory alternative that still utilizes validity information, distinguishing it from both Tallying and TTB."
}
```

## Usage

```json
{
  "prompt_token_count": 4508,
  "candidates_token_count": 273,
  "total_token_count": 5182
}
```
