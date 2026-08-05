# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_4" and "pi_3") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_4" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_3" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_4
Context-Dependent Probabilistic Strategy Selection (Take-the-First, else Tally) with parameterized depth decay. Decision-makers probabilistically switch between a non-compensatory heuristic (TTB) and a compensatory strategy (Tallying) based on choice difficulty (depth of the first discriminating cue). The probability of using TTB starts at a baseline 'alpha' for depth=0 and decays by 'gamma' for deeper cues. Tallying predictions are softened by a temperature parameter.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Determine TTB prediction and the depth of the first discriminating cue
    cue_order = np.argsort(-val, kind="stable").tolist()
    first_disc_idx = -1
    winner_ttb = -1
    for i, j in enumerate(cue_order):
        if a[j] > b[j]:
            winner_ttb = 0
            first_disc_idx = i
            break
        if b[j] > a[j]:
            winner_ttb = 1
            first_disc_idx = i
            break
            
    if winner_ttb == 0:
        p_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        p_ttb = np.array([0.0, 1.0])
    else:
        p_ttb = np.array([0.5, 0.5])
        
    # Determine Tallying prediction using softmax
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])
    tau = float(parameters["tau"])
    
    z = tau * (scores - np.max(scores))
    e = np.exp(z)
    p_tally = e / np.sum(e)
        
    # Probabilistic strategy switch based on choice difficulty (depth of first discriminating cue)
    gamma = float(parameters["gamma"])
    alpha = float(parameters["alpha"])
    if first_disc_idx == -1:
        prob_ttb = 0.0
    else:
        prob_ttb = alpha * (gamma ** first_disc_idx)
        
    p_core = prob_ttb * p_ttb + (1.0 - prob_ttb) * p_tally
    
    # Independent lapse rate
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_3
People evaluate options using a non-linearly Weighted Additive (WADD) strategy. Each option's value is the sum of its features weighted by their perceived importance, which is a non-linear power function of the objective cue validities. This allows a single high-validity cue to balance out multiple lower-validity cues, resulting in compensatory trade-offs and choice probabilities near 0.5 on conflict trials.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Compute weighted sum of features for each option
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.75, 0.7, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[1, 0, 1, 1, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[1, 1, 0, 1, 1, 0]
  trial 4: A=[1, 1, 1, 1, 0, 0]  B=[1, 1, 1, 0, 1, 1]
  trial 5: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 7: A=[1, 1, 0, 1, 1, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0, 1, 1]  B=[1, 1, 1, 1, 0, 0]

**Rationale:** This experiment specifically targets the 'depth of the first discriminating cue' mechanism unique to Theory 1. Theory 1 posits that subjects probabilistically switch from Take-The-Best (TTB) to Tallying as the first discriminating cue is found deeper in the cue hierarchy (via the gamma decay parameter). To test this, we design trial pairs where TTB and Tallying consistently oppose each other (e.g., TTB favors one option based on the highest available discriminating cue, while Tallying favors the other option based on a simple majority of lower-validity cues). Across trials, we systematically increase the depth of the first discriminating cue (from depth 0 to depth 3) by tying the top N cues. Theory 1 predicts a sharp, depth-dependent decay in the probability of choosing the TTB-favored option as the decision strategy falls back to Tallying. Theory 2 (WADD) evaluates options based on a continuous weighted sum; while the absolute difference in weighted sums may vary slightly across these trials, WADD does not contain a discrete depth-based penalty, leading to a fundamentally different pattern of choice probabilities across the depth gradient.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the difference in the rate of Take-The-Best (TTB) consistent choices between trials where the first discriminating cue is at depth 0 versus depth 3. Theory 1 predicts a sharp, explicit decay in TTB usage as depth increases (due to its gamma parameter), leading to a large positive difference. Theory 2 (WADD) evaluates options continuously without a discrete switch, so its choice probabilities will behave differently across the depth gradient.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    d0_ttb = []
    d3_ttb = []
    
    for a, b, resp in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] != b[i]:
                ttb_choice = 0 if a[i] > b[i] else 1
                is_ttb = 1 if resp == ttb_choice else 0
                if i == 0:
                    d0_ttb.append(is_ttb)
                elif i == 3:
                    d3_ttb.append(is_ttb)
                break
                
    m0 = np.mean(d0_ttb) if len(d0_ttb) > 0 else 0.0
    m3 = np.mean(d3_ttb) if len(d3_ttb) > 0 else 0.0
    
    return float(m0 - m3)

### RESULTS
- Predicted under pi_4 (simulated): 0.4142 (var=0.0591)
- Predicted under pi_3 (simulated): -0.0567 (var=0.0265)
- Observed on real data: 0.0000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 0]
  trial 3: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 6: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 8: A=[0, 1, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** This design systematically varies the depth of the first discriminating cue while maintaining a consistent conflict between the highest-validity discriminating cue and the simple majority of lower-validity cues. Theory 2 (Context-Dependent Strategy Selection) predicts a discrete, sharp drop in the probability of choosing the TTB-favored option as the depth of the first discriminating cue increases, because it explicitly falls back to Tallying (which favors the other option). Theory 1 (WADD) computes a continuous non-linear weighted sum; it lacks a discrete depth penalty and will instead show choice probabilities that track the continuous differences in weighted evidence, providing a clear quantitative dissociation from Theory 2's depth-decay mechanism.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric exploits a structural symmetry in the Competing Theory (Context-Dependent Strategy Selection) that WADD strictly breaks. In both Trial 1 and Trial 4, the first discriminating cue is at depth 0, meaning the probability of using TTB is identical. Furthermore, in both trials, Option A has exactly 1 fewer positive feature than Option B, meaning the Tallying strategy's softmax probability for Option A is also mathematically identical (since it depends only on the difference in sums, which is -1 in both cases). Thus, the Competing Theory predicts exactly the same probability of choosing A for both trials, yielding an expected difference of 0. In contrast, WADD integrates the non-linear weights of all features. The difference in weighted evidence for Trial 1 vs Trial 4 is rigorously non-zero (specifically, 2*(0.7^g) - 0.6^g - 0.5^g > 0), so WADD reliably predicts a higher probability of choosing A in Trial 1 than in Trial 4.

Source:
def metric(data: pd.DataFrame) -> float:
    # Identify Trial 1 and Trial 4 by their unique Option A profiles
    t1_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 0))
    t4_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    
    return float(p_a_t1 - p_a_t4)

### RESULTS
- Predicted under pi_4 (simulated): -0.0067 (var=0.0275)
- Predicted under pi_3 (simulated): 0.1950 (var=0.0591)
- Observed on real data: 0.0000 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[0, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 0]  B=[1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract ratings into 2D arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # TTB prediction: first discriminating feature in descending validity order
    ttb_preds = np.full(len(data), -1)
    for i in range(a_ratings.shape[1]):
        mask_a = (a_ratings[:, i] > b_ratings[:, i]) & (ttb_preds == -1)
        mask_b = (b_ratings[:, i] > a_ratings[:, i]) & (ttb_preds == -1)
        ttb_preds[mask_a] = 0
        ttb_preds[mask_b] = 1
        
    # Tally prediction: sum of feature-wise wins
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    tally_preds = np.full(len(data), -1)
    tally_preds[a_wins > b_wins] = 0
    tally_preds[b_wins > a_wins] = 1
    
    # Identify trials where TTB and Tallying make opposite predictions
    disagree_mask = (ttb_preds != -1) & (tally_preds != -1) & (ttb_preds != tally_preds)
    
    if not np.any(disagree_mask):
        return 0.5
        
    # Calculate proportion of choices aligning with TTB on these crucial trials
    responses = data['response'].values
    matches = (responses[disagree_mask] == ttb_preds[disagree_mask])
    
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5360 (var=0.0087)
**Predicted under pi_4:** 0.6250 (var=0.0133)
**Predicted under pi_3:** 0.5487 (var=0.0539)

### Experiment 4
**Design**
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[1, 1, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    a_wins = np.sum(a_mat > b_mat, axis=1)
    b_wins = np.sum(b_mat > a_mat, axis=1)
    
    tally_pref = np.where(a_wins > b_wins, 0, np.where(b_wins > a_wins, 1, -1))
    
    diff = a_mat - b_mat
    nz = diff != 0
    has_nz = np.any(nz, axis=1)
    first_nz_idx = np.argmax(nz, axis=1)
    first_diff = diff[np.arange(len(diff)), first_nz_idx]
    
    ttb_pref = np.where(has_nz, np.where(first_diff > 0, 0, 1), -1)
    
    mask = (tally_pref != -1) & (ttb_pref != -1) & (tally_pref != ttb_pref)
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values[mask]
    tally_prefs = tally_pref[mask]
    
    return float(np.mean(responses == tally_prefs))
```

**Observed (real) value:** 0.5200 (var=0.0096)
**Predicted under pi_4:** 0.4938 (var=0.0156)
**Predicted under pi_3:** 0.4533 (var=0.0339)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  A=[1, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.vstack(data['option_a_ratings'].values)
    b_mat = np.vstack(data['option_b_ratings'].values)
    
    # Difference in features
    diff = a_mat - b_mat
    
    # Find the first discriminating cue (TTB logic)
    is_diff = diff != 0
    has_diff = is_diff.any(axis=1)
    first_diff_idx = is_diff.argmax(axis=1)
    first_diff_val = diff[np.arange(len(diff)), first_diff_idx]
    ttb_pred = np.where(first_diff_val > 0, 0, 1)
    
    # WADD logic (with gamma=1)
    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])
    wadd_diff = diff @ val
    wadd_pred = np.where(wadd_diff > 0, 0, 1)
    
    # Identify conflict trials where TTB and WADD (gamma=1) disagree
    conflict = (ttb_pred != wadd_pred) & has_diff
    
    if not conflict.any():
        return 0.5
        
    # Calculate proportion of choices on conflict trials that align with TTB
    responses = data['response'].values
    matches = (responses[conflict] == ttb_pred[conflict]).sum()
    
    return float(matches / conflict.sum())
```

**Observed (real) value:** 0.4222 (var=0.1296)
**Predicted under pi_4:** 0.6064 (var=0.0212)
**Predicted under pi_3:** 0.5472 (var=0.0395)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    diff = sum_a - sum_b
    choose_a = 1.0 - data['response']
    cov = np.cov(diff, choose_a)[0, 1]
    return float(cov)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_4:** 0.1818 (var=0.0465)
**Predicted under pi_3:** 0.5299 (var=0.1410)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across the experiments, both theories show significant predictive failures. In Experiment 1, Theory 1 predicted a large positive difference (0.41) due to its core depth-decay mechanism, but the observed value was exactly 0.0000, indicating that the depth of the first discriminating cue does not cause a fallback to Tallying. In Experiment 2, Theory 2 (WADD) predicted a difference of 0.1950 based on continuous evidence integration, but the observed value was again 0.0000, suggesting that subjects are not integrating all cues in a weighted additive manner, but rather relying on a strategy that preserves the structural symmetries of the trial pairs (like pure TTB or a fixed Tallying mixture). In Experiment 6, the observed covariance between tally differences and choices was 0.0000, further discrediting WADD which predicted 0.5299. Since Theory 1's defining feature (depth-dependent strategy switching) is directly contradicted by Experiment 1, the theory is fundamentally flawed.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a 'Constant-Mixture Strategy Selection' theory. Unlike Theory 1, which posits that the probability of using Take-The-Best (TTB) decays as the depth of the first discriminating cue increases, this new theory should propose that subjects have a fixed, individual-specific probability of using TTB versus Tallying on any given trial, completely independent of cue depth. This explains why the depth manipulation in Experiment 1 yielded a 0.0 difference, while preserving the structural symmetry in Experiment 2 (yielding a 0.0 difference) that WADD failed to capture."
}
```

## Usage

```json
{
  "prompt_token_count": 7412,
  "candidates_token_count": 385,
  "total_token_count": 8950
}
```
