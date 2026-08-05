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


## THEORY 2 — pi_4
Non-linear Subjective Weighting Model: Subjects evaluate options by computing a weighted sum of their features, but they do not use the objective cue validities directly. Instead, subjective cue weights are a power function of the provided validities. An individual-specific exponent parameter controls the non-linearity of this transformation. This single mechanism unifies multiple decision strategies: an exponent near 0 flattens the weights (yielding Equal-Weight/Tallying), an exponent of 1 uses the validities linearly (yielding WADD), and a large exponent strongly amplifies the most valid cues (yielding non-compensatory Take The Best behavior).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear subjective weighting: w_i = v_i ^ gamma
    # Validities are in [0.5, 1.0], so base is positive.
    w = np.maximum(validities, 0.0) ** gamma
    
    # Option scores are the weighted sum of features
    scores = stim @ w
    
    # Softmax over scores with inverse temperature beta
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Add independent lapse noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Probabilistic Heuristic Selection (PHS) from the Non-linear Subjective Weighting Model (NLSW) by exploiting PHS's insensitivity to the specific validities of secondary cues. Under PHS, the choice probability is a mixture of Take The Best (TTB) and Tallying. If we hold the TTB winner constant (e.g., Option A always has the single highest validity cue) and hold the Tallying winner constant (e.g., Option B always has a greater total number of positive cues), PHS predicts the exact same choice probability across trials, regardless of which specific lower-validity cues Option B possesses. By contrast, NLSW computes a weighted sum using all validities. By shifting Option B's positive features from higher-validity cues to lower-validity cues across trials while keeping the total count identical, NLSW predicts a systematic shift in choice probabilities, whereas PHS predicts strict invariance.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the within-subject difference in the likelihood of choosing the Take The Best (TTB) winner (the option with the highest validity cue) between 'weak compensatory' trials and 'strong compensatory' trials. In 'weak compensatory' trials, the alternative option has fewer, less valid cues (e.g., cues 4 and 5), whereas in 'strong compensatory' trials, it has more valid cues (e.g., cues 2, 3, and 4). Under the Probabilistic Heuristic Selection (PHS) model, because the TTB winner and Tallying winner are identical across these trials, the choice probability remains invariant, predicting a difference near zero. Under the Non-linear Subjective Weighting Model (NLSW), the subjective value of the compensatory option shifts significantly, predicting a much higher TTB choice rate in weak compensatory trials than in strong ones.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def get_trial_type(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        if a[0] == 1:
            other = b
        else:
            other = a
            
        if other == (0, 1, 1, 1, 0):
            return 'strong'
        elif other == (0, 0, 0, 1, 1):
            return 'weak'
        return 'other'

    data['trial_type'] = data.apply(get_trial_type, axis=1)
    
    def is_ttb_choice(row):
        a_has_cue = row['option_a_ratings'][0] == 1
        return 1 if (row['response'] == 0 and a_has_cue) or (row['response'] == 1 and not a_has_cue) else 0

    data['ttb_choice'] = data.apply(is_ttb_choice, axis=1)
    
    subj_diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        strong_rate = subj_df[subj_df['trial_type'] == 'strong']['ttb_choice'].mean()
        weak_rate = subj_df[subj_df['trial_type'] == 'weak']['ttb_choice'].mean()
        if pd.notna(strong_rate) and pd.notna(weak_rate):
            subj_diffs.append(weak_rate - strong_rate)
            
    return float(np.mean(subj_diffs)) if subj_diffs else 0.0

### RESULTS
- Predicted under pi_3 (simulated): 0.0433 (var=0.0126)
- Predicted under pi_4 (simulated): 0.2833 (var=0.0876)
- Observed on real data: 0.4767 (var=0.1713)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=8):** [0.95, 0.9, 0.85, 0.8, 0.75, 0.6, 0.6, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 0, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 0, 1, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 0, 0, 1, 0]
  trial 8: A=[1, 0, 0, 0, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0, 0, 0, 0]

**Rationale:** This design introduces a profound qualitative dissociation by creating 'Double-Agreement Override' trials. In these critical trials, Option A possesses the single highest-validity cue (winning Take The Best) AND a strictly greater total number of positive features (winning Tallying). Because the Probabilistic Heuristic Selection (PHS) model is a mixture of TTB and Tallying, it must overwhelmingly predict Option A on these trials, regardless of the mixing parameter. However, the Non-linear Subjective Weighting (NLSW) model allows for intermediate non-linear scaling of validities (e.g., a gamma exponent between 2 and 5). Under this intermediate non-linear regime, a coalition of high-to-medium validity cues in Option B mathematically outweighs Option A's single highest-validity cue and its numerous low-validity cues. Thus, NLSW uniquely predicts that subjects can prefer Option B despite both TTB and Tallying pointing to Option A, providing a strict test between the models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of 'override' choices on trials where both Take The Best (TTB) and Tallying agree on the same option. The Probabilistic Heuristic Selection (PHS) model, which mixes TTB and Tallying, will overwhelmingly choose the agreed-upon option, resulting in an override rate near zero (driven only by the lapse rate). In contrast, the Non-linear Subjective Weighting (NLSW) model can assign intermediate convex weights to the cues, meaning a coalition of medium-validity cues can mathematically outweigh a single highest-validity cue plus several low-validity cues, leading to a much higher rate of choosing the 'override' option.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    overrides = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        ttb_winner = None
        for i in range(len(a)):
            if a[i] > b[i]:
                ttb_winner = 0
                break
            elif b[i] > a[i]:
                ttb_winner = 1
                break
                
        sum_a = sum(a)
        sum_b = sum(b)
        tally_winner = 0 if sum_a > sum_b else (1 if sum_b > sum_a else None)
        
        if ttb_winner is not None and tally_winner is not None and ttb_winner == tally_winner:
            chosen = row['response']
            if chosen != ttb_winner:
                overrides.append(1)
            else:
                overrides.append(0)
                
    if not overrides:
        return 0.0
    return float(np.mean(overrides))

### RESULTS
- Predicted under pi_3 (simulated): 0.0537 (var=0.0015)
- Predicted under pi_4 (simulated): 0.3604 (var=0.0143)
- Observed on real data: 0.4729 (var=0.0463)

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
**Predicted under pi_4:** 0.5029 (var=0.0626)

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
**Predicted under pi_4:** 0.4823 (var=0.0320)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 1]  B=[0, 0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 0, 0, 0]  B=[1, 1, 0, 0, 1, 1, 1]

**Metric**
```python
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

```

**Observed (real) value:** 0.2317 (var=0.0651)
**Predicted under pi_3:** 0.8050 (var=0.0118)
**Predicted under pi_4:** 0.2029 (var=0.0250)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 1, 1, 1, 0]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 1, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 0, 1, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 1, 1, 0, 1]  B=[0, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1, 0, 0]
  A=[0, 0, 0, 0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0, 1, 1]
  A=[0, 0, 1, 1, 0, 0, 1, 1]  B=[1, 1, 0, 0, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    a_0 = data['option_a_ratings'].apply(lambda x: x[0])
    b_1 = data['option_b_ratings'].apply(lambda x: x[1])
    critical_mask = (a_0 == 1) & (b_1 == 1)
    if critical_mask.sum() == 0:
        return 0.5
    return float(data.loc[critical_mask, 'response'].mean())
```

**Observed (real) value:** 0.8158 (var=0.0450)
**Predicted under pi_3:** 0.0650 (var=0.0015)
**Predicted under pi_4:** 0.7850 (var=0.0505)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the six experiments, Theory 2 (Non-linear Subjective Weighting, pi_4) consistently outperforms Theory 1 (Probabilistic Heuristic Selection, pi_3), particularly on critical dissociation tasks. In Experiments 1, 2, 5, and 6, Theory 1 fails catastrophically because it assumes subjects rigidly follow a mixture of Take The Best and Tallying, making it incapable of explaining 'Double-Agreement Overrides' or sensitivity to secondary cue validities. Theory 2 correctly captures these phenomena by using a compensatory weighted sum with non-linear validity scaling. However, Theory 2 slightly misses the exact observed rates in some experiments (e.g., Exps 3 and 4), suggesting there is room for a competing theory that captures these non-linear, compensatory choice patterns without relying strictly on the exact numerical values of the validities (which Theory 2 uses as the base of its power function). Theory 1 is fundamentally degenerate and should be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Rank-Based Weighting' theory. Instead of using the raw objective validities or transforming them via a power function (as in NLSW), subjects might rank the features based on their provided validities and assign subjective weights that decay as a function of their ordinal rank (e.g., exponential or inverse rank decay). This theory would remain compensatory and allow for overrides (similar to NLSW) but would predict that the exact numerical differences between validities matter less than their ordered priority. This provides a strong, plausible alternative to NLSW that dispenses with the raw validity values entirely."
}
```

## Usage

```json
{
  "prompt_token_count": 7100,
  "candidates_token_count": 367,
  "total_token_count": 8123
}
```
