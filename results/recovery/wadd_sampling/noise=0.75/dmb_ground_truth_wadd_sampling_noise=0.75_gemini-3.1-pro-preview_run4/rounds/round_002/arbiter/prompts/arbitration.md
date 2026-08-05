# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_4") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_4" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
People evaluate options by computing a comprehensive weighted sum of all available features. Each feature's contribution is proportional to its objective validity transformed into log-odds (log-likelihood ratios). This allows the compensatory mechanism to properly weigh probabilistic evidence, preventing multiple weak cues from inappropriately dominating a single strong cue. Choices are generated probabilistically by applying a softmax function over the options' weighted scores, subject to an independent lapse rate.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    # Transform validities into log-odds to correctly accumulate probabilistic evidence
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute weighted sum of features for each option
    scores = np.dot(stim, w)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
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
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_4
Strategy Mixture Theory with Independent Scaling: Individuals use a probabilistic mixture of distinct heuristics (WADD, Tallying, and Take-The-Best), but because the internal evidence scales of these heuristics vary dramatically (log-odds sums vs. integer counts vs. binary indicators), each heuristic applies its own independent temperature parameter to properly calibrate its choice probabilities before mixing.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    scores_wadd = np.dot(stim, w)
    
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_tally = np.array([a_wins, b_wins])
    
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
    if winner is None:
        scores_ttb = np.array([0.0, 0.0])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
        
    beta_wadd = float(parameters["beta_wadd"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    
    def get_probs(scores, beta):
        z = beta * (scores - np.max(scores))
        e = np.exp(z)
        return e / np.sum(e)
        
    p_wadd = get_probs(scores_wadd, beta_wadd)
    p_tally = get_probs(scores_tally, beta_tally)
    p_ttb = get_probs(scores_ttb, beta_ttb)
    
    w1 = float(parameters["w_wadd"])
    w2 = float(parameters["w_tally"])
    w3 = float(parameters["w_ttb"])
    w_sum = w1 + w2 + w3 + 1e-9
    
    p_mix = (w1 * p_wadd + w2 * p_tally + w3 * p_ttb) / w_sum
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)

## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.55, 0.52, 0.51]

**Trial pairs (n=5):**
  trial 1: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]

**Rationale:** To dissociate WADD from a mixture of Tallying and Take-The-Best (TTB), we construct critical trials where WADD predicts the opposite of both TTB and Tallying. By setting one extremely high validity cue (0.95), two highly valid cues (0.90, 0.85), and three weak cues (~0.5), we can create a situation where Option A wins on the two highly valid cues, while Option B wins on the extremely high validity cue AND the three weak cues. Here, WADD's log-odds integration strongly favors Option A. However, TTB favors Option B (due to the highest validity cue), and Tallying also favors Option B (4 winning cues vs 2). A mixture model will thus predict a much higher choice probability for Option B than the pure WADD model. We include symmetric versions and trials where heuristics partially agree to estimate mixture weights.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices consistent with the WADD model on the two critical trials (Trial 1 and Trial 2). On these specific trials, the WADD model's log-odds integration favors an option that is opposed by BOTH the Take-The-Best (TTB) heuristic (which favors the option with the highest validity cue) and the Tallying heuristic (which favors the option with the highest number of positive cues). The pure WADD theory predicts a high probability of choosing the WADD-favored option, whereas the Strategy Mixture Theory predicts a significantly lower probability because the TTB and Tallying components of the mixture will strongly pull the choice in the opposite direction.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    
    # Trial 1: WADD prefers A, Tallying and TTB prefer B
    is_t1 = a_tuples == (0, 1, 1, 0, 0, 0)
    
    # Trial 2: WADD prefers B, Tallying and TTB prefer A
    is_t2 = a_tuples == (1, 0, 0, 1, 1, 1)
    
    wadd_choice_t1 = (data.loc[is_t1, 'response'] == 0).astype(float)
    wadd_choice_t2 = (data.loc[is_t2, 'response'] == 1).astype(float)
    
    combined = np.concatenate([wadd_choice_t1.values, wadd_choice_t2.values])
    if len(combined) == 0:
        return 0.5
    return float(np.mean(combined))

### RESULTS
- Predicted under pi_3 (simulated): 0.8463 (var=0.0087)
- Predicted under pi_4 (simulated): 0.3784 (var=0.0173)
- Observed on real data: 0.4568 (var=0.0095)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.8, 0.75, 0.7, 0.6]

**Trial pairs (n=12):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 9: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 10: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 11: A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 0, 0]
  trial 12: A=[0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Strategy Mixture Theory from a pure Weighted Additive (WADD) model, we design a 5-feature experiment with validities that create distinct profiles of agreement and disagreement among WADD, Take-The-Best (TTB), and Tallying. In a pure WADD model, choice probabilities are exclusively determined by the difference in the options' log-odds sums. In contrast, the Mixture Theory incorporates TTB (which is highly sensitive to the single most valid discriminating cue) and Tallying (which is sensitive only to the unweighted count of winning cues), each with independent softmax temperatures. By including trials where WADD prefers one option while TTB and Tallying either tie or pull in opposite directions, the Mixture Theory predicts choice probabilities that systematically deviate from the strict log-odds differences predicted by pure WADD.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of choices aligned with the Take-The-Best (TTB) heuristic specifically on trials where TTB and the pure Weighted Additive (WADD) model disagree. In pure WADD, choices that contradict the weighted log-odds sum should be selected less than 50% of the time. However, under the Strategy Mixture Theory, the presence of a dedicated TTB heuristic component will significantly elevate the selection rate of the option favored by the most valid cue, leading to a much higher proportion of TTB-aligned choices on these critical conflict trials.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    val = np.array([0.95, 0.8, 0.75, 0.7, 0.6])
    w = np.log(val / (1.0 - val))
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    
    score_a = a_ratings.dot(w)
    score_b = b_ratings.dot(w)
    
    # TTB prefers A, but WADD prefers B
    cond1 = (a_ratings[:, 0] > b_ratings[:, 0]) & (score_a < score_b)
    # TTB prefers B, but WADD prefers A
    cond2 = (b_ratings[:, 0] > a_ratings[:, 0]) & (score_b < score_a)
    
    responses = data['response'].values
    
    ttb_chose_a = cond1 & (responses == 0)
    ttb_chose_b = cond2 & (responses == 1)
    
    ttb_choices = np.sum(ttb_chose_a) + np.sum(ttb_chose_b)
    total_disagreements = np.sum(cond1) + np.sum(cond2)
    
    if total_disagreements == 0:
        return 0.0
        
    return float(ttb_choices / total_disagreements)


### RESULTS
- Predicted under pi_3 (simulated): 0.1913 (var=0.0139)
- Predicted under pi_4 (simulated): 0.3856 (var=0.0216)
- Observed on real data: 0.3875 (var=0.0057)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    diff = a_ratings - b_ratings
    
    # TTB: winner based on the first discriminating cue
    first_diff_idx = np.argmax(diff != 0, axis=1)
    ttb_winner = np.where(diff[np.arange(len(diff)), first_diff_idx] > 0, 0, 1)
    
    # Tallying: winner based on total number of winning cues
    a_wins = np.sum(diff > 0, axis=1)
    b_wins = np.sum(diff < 0, axis=1)
    
    # Identify conflict trials where Tallying has a strict winner that disagrees with TTB
    tally_winner = np.where(a_wins > b_wins, 0, 1)
    conflict_mask = (a_wins != b_wins) & (ttb_winner != tally_winner)
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    ttb_consistent = (responses[conflict_mask] == ttb_winner[conflict_mask])
    
    return float(np.mean(ttb_consistent))
```

**Observed (real) value:** 0.4619 (var=0.0040)
**Predicted under pi_3:** 0.5666 (var=0.0056)
**Predicted under pi_4:** 0.5231 (var=0.0144)

### Experiment 4
**Design**
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    mask = a_wins != b_wins
    if not np.any(mask):
        return 0.5
        
    tally_pred = (b_wins[mask] > a_wins[mask]).astype(int)
    return float(np.mean(tally_pred == data['response'].values[mask]))
```

**Observed (real) value:** 0.5637 (var=0.0028)
**Predicted under pi_3:** 0.7028 (var=0.0062)
**Predicted under pi_4:** 0.6109 (var=0.0083)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    # Log-odds weights for validities: [0.9, 0.8, 0.6, 0.55, 0.51]
    w = np.array([2.19722458, 1.38629436, 0.40546511, 0.2006707 , 0.04000533])
    
    def wadd_predicts_A(row):
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        return np.sum(a * w) > np.sum(b * w)
    
    wadd_pred_A = data.apply(wadd_predicts_A, axis=1)
    
    # Subject chose A if response == 0, B if response == 1
    match = (wadd_pred_A & (data['response'] == 0)) | (~wadd_pred_A & (data['response'] == 1))
    
    return float(match.mean())
```

**Observed (real) value:** 0.4629 (var=0.0047)
**Predicted under pi_3:** 0.8102 (var=0.0128)
**Predicted under pi_4:** 0.5973 (var=0.0174)

### Experiment 6
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Convert lists to tuples to allow for equality comparison
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Identify the critical trials where Tallying and WADD make opposing predictions
    t1_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 0, 0, 0, 0))
    t5_mask = (a_tuples == (0, 1, 1, 1, 0)) & (b_tuples == (1, 0, 0, 0, 1))
    t3_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    
    mask_all = t1_mask | t5_mask | t3_mask
    if not mask_all.any():
        return 0.5
        
    # Tallying predictions: 
    # T1: A wins on 3 features, B on 1 -> prefers A (0)
    # T5: A wins on 3 features, B on 2 -> prefers A (0)
    # T3: A wins on 2 features, B on 3 -> prefers B (1)
    preds = pd.Series(index=data.index, data=np.nan)
    preds.loc[t1_mask] = 0
    preds.loc[t5_mask] = 0
    preds.loc[t3_mask] = 1
    
    # Calculate the proportion of choices matching Tallying's predictions
    match = (data.loc[mask_all, 'response'] == preds.loc[mask_all])
    return float(match.mean())
```

**Observed (real) value:** 0.5211 (var=0.0132)
**Predicted under pi_3:** 0.1539 (var=0.0112)
**Predicted under pi_4:** 0.3950 (var=0.0238)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, THEORY 2 (pi_4, Strategy Mixture) consistently outperforms THEORY 1 (pi_3, Pure WADD with log-odds). THEORY 1 makes overly extreme predictions (e.g., 0.84, 0.19, 0.81, 0.15) because the log-odds transformation assigns massive weights to highly valid cues, whereas human choices in these tasks consistently hover much closer to 0.5 (ranging from 0.38 to 0.56). This indicates that human decision-making in this paradigm is significantly noisier or less strictly compensatory than a normative log-odds integration would suggest. While THEORY 2 captures the data better by mixing in Tallying and TTB, its mixture of three separate heuristics with independent temperatures is highly parameterized. THEORY 1 is degenerate because its core assumption of log-odds integration fundamentally mischaracterizes human cue weighting in this domain.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Replace THEORY 1 with a 'Rank-based Weighting Theory'. Instead of transforming validities into log-odds (which causes extreme predictions), propose that subjects weight cues based simply on their ordinal rank (e.g., weights proportional to 1/rank, or linearly decreasing). This maintains a single, unified compensatory mechanism (unlike the mixture model of THEORY 2) but naturally dampens the dominance of highly valid cues, producing softer choice probabilities that should better align with the observed data clustering around 0.45-0.55."
}
```

## Usage

```json
{
  "prompt_token_count": 7176,
  "candidates_token_count": 347,
  "total_token_count": 8858
}
```
