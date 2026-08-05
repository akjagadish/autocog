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
People are heterogeneous in their decision-making strategies, with some choices driven by a non-compensatory heuristic (Take The Best) and others by a compensatory strategy (Weighted Additive). The population consists of individuals who employ a mixture of these strategies, governed by a subjective mixture weight. By blending a frugal, single-reason strategy with a fully compensatory evaluation, the model captures both the variance and the balanced aggregate behavior observed across decision-making experiments.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    n_features = stim.shape[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    # --- TTB (Take The Best) ---
    cue_order = np.argsort(-validities, kind="stable").tolist()
    a, b = stim[0], stim[1]
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    beta = float(parameters["beta"])
    
    if winner_ttb is None:
        p_ttb = np.ones(2) / 2.0
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * scores_ttb
        e_ttb = np.exp(z_ttb - np.max(z_ttb))
        p_ttb = e_ttb / e_ttb.sum()
        
    # --- WADD (Weighted Additive) ---
    scores_wadd = stim @ (validities * w)
    z_wadd = beta * scores_wadd
    e_wadd = np.exp(z_wadd - np.max(z_wadd))
    p_wadd = e_wadd / e_wadd.sum()
    
    # --- MIXTURE ---
    wadd_prob = float(parameters["wadd_prob"])
    p_core = wadd_prob * p_wadd + (1.0 - wadd_prob) * p_ttb
    
    epsilon = float(parameters["epsilon"])
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(2) / 2.0)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## THEORY 2 — pi_4
Equal Weights with High Lapse: Subjects often find the integration of complex cue validities too cognitively demanding or disengaging in these conflict paradigms. As a result, they ignore the provided validities entirely and fall back on a simple 'Tallying' (Equal Weights) heuristic, where they just count the number of positive features for each option. Furthermore, due to the high cognitive friction or confusion, subjects exhibit a very high baseline guessing rate (lapse), meaning that on the vast majority of trials they simply guess randomly. This explains why behavior across various conflict and agreement metrics hovers so closely to 0.5 or 0 difference.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    
    # Tallying: equal weights for all features (counting positive cues)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tallied scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # High uniform lapse blended in
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.95, 0.85, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 6: A=[0, 1, 0, 0]  B=[1, 0, 1, 1]

**Rationale:** To dissociate the Strategy Mixture model (Advocated) from the Equal Weights with High Lapse model (Competing), we exploit the fact that the Competing theory relies exclusively on 'Tallying' (counting positive cues) heavily masked by a high guessing rate (lapse >= 0.8). By designing trials where Tallying predicts a dead heat (equal number of positive cues for both options) but the validities strongly favor one option, the Competing theory is forced to predict exactly a 50/50 choice probability. In contrast, the Advocated theory will predict a strong preference for the option favored by the high-validity cues, driven by both Take-The-Best (TTB) and Weighted Additive (WADD) components. Additionally, we include trials where Tallying strongly opposes the highest-validity cue to further contrast the high-lapse equal-weighting predictions against the structured TTB/WADD mixture.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject chooses the option favored by the most valid cue (cue 1, validity 0.95). In the Advocated theory, the mixture of Take-The-Best (which strictly follows cue 1) and Weighted Additive (which heavily weights cue 1) will produce a high alignment rate. In contrast, the Competing theory relies on Equal Weights (Tallying) combined with a very high guessing rate (lapse >= 0.8). Since Tallying often ties or even opposes cue 1 in this specific experimental design, and the high lapse rate dilutes any systematic preference, the Competing theory will predict an alignment rate very close to or slightly below 0.5. This creates a strong and consistent divergence between the two theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract the first cue (highest validity) for options A and B
    a_cue1 = data['option_a_ratings'].apply(lambda x: x[0])
    b_cue1 = data['option_b_ratings'].apply(lambda x: x[0])
    
    # Determine if the chosen option had a 1 on the most valid cue
    # response == 0 means A was chosen, response == 1 means B was chosen
    chosen_cue1 = np.where(data['response'] == 0, a_cue1, b_cue1)
    
    # Return the proportion of trials where the choice aligned with the most valid cue
    return float(np.mean(chosen_cue1))

### RESULTS
- Predicted under pi_3 (simulated): 0.6104 (var=0.0132)
- Predicted under pi_4 (simulated): 0.4892 (var=0.0028)
- Observed on real data: 0.4996 (var=0.0028)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.55, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 4: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated theory (Equal Weights with High Lapse) from the Competing theory (Strategy Mixture of TTB and WADD), this design creates 'anti-validity' trials where the sheer number of positive cues opposes the highest-validity cues. For example, in Trial 1, Option A is supported by only two cues, but they are the most valid ones (0.95 and 0.90), whereas Option B is supported by three low-validity cues (0.55, 0.55, 0.50). The Advocated theory, which relies on tallying (Equal Weights) heavily masked by guessing, predicts a weak preference for Option B (since 3 > 2). Conversely, the Competing theory predicts a strong preference for Option A, because both TTB (relying on the 0.95 cue) and WADD (1.85 vs 1.60) strongly favor A. Additional trials systematically vary whether EQW predicts a tie while TTB/WADD predict a clear winner, providing a comprehensive test of whether subjects are entirely validity-insensitive as the Advocated theory claims.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times subjects choose Option A in Trial 1, where Option A has fewer positive cues (2) but they are the most valid, while Option B has more positive cues (3) but they are less valid. The Advocated theory (Equal Weights + High Lapse) predicts subjects will slightly favor Option B (since 3 > 2) or be completely indifferent due to high lapsing, resulting in a metric value <= 0.5. The Competing theory (TTB + WADD) predicts a strong preference for Option A, because both Take-The-Best (relying on the 0.95 cue) and Weighted Additive (1.85 vs 1.60) strongly favor A, resulting in a metric value > 0.5. This clear opposition in choice direction will yield a highly significant discrimination.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    def is_trial_1(row):
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        return a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1)
        
    mask = data.apply(is_trial_1, axis=1)
    if not mask.any():
        return 0.5
        
    return float(np.mean(data.loc[mask, 'response'] == 0))

### RESULTS
- Predicted under pi_3 (simulated): 0.6225 (var=0.0523)
- Predicted under pi_4 (simulated): 0.4662 (var=0.0191)
- Observed on real data: 0.5350 (var=0.0082)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    A = np.stack(data['option_a_ratings'].values)
    B = np.stack(data['option_b_ratings'].values)
    
    diff = A - B
    
    ttb_preds = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(A.shape[1]):
            if diff[i, j] == 1:
                ttb_preds[i] = 0
                break
            elif diff[i, j] == -1:
                ttb_preds[i] = 1
                break
                
    matches = (data['response'].values == ttb_preds)
    return float(np.mean(matches))
```

**Observed (real) value:** 0.5025 (var=0.0026)
**Predicted under pi_3:** 0.4948 (var=0.0127)
**Predicted under pi_4:** 0.4506 (var=0.0030)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
```

**Observed (real) value:** 0.4996 (var=0.0028)
**Predicted under pi_3:** 0.5142 (var=0.0127)
**Predicted under pi_4:** 0.5415 (var=0.0036)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  A=[0, 0, 1, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_chosen = []
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        a_f0 = a[0]
        b_f0 = b[0]
        
        # Focus on conflict trials where the option with the best cue (f0) 
        # has very few other positive cues, while the other option has many.
        if a_f0 == 1 and b_f0 == 0:
            if sum(a) <= 2 and sum(b) >= 4:
                ttb_chosen.append(1 if resp == 0 else 0)
        elif b_f0 == 1 and a_f0 == 0:
            if sum(b) <= 2 and sum(a) >= 4:
                ttb_chosen.append(1 if resp == 1 else 0)
                
    if not ttb_chosen:
        return 0.5
    return float(np.mean(ttb_chosen))
```

**Observed (real) value:** 0.4947 (var=0.0048)
**Predicted under pi_3:** 0.4603 (var=0.0125)
**Predicted under pi_4:** 0.4503 (var=0.0051)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    data = data.copy()
    # Safely convert list of ratings to string for easy matching
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
    
    # Agreement trials: WADD and TTB both strongly favor the '11111' option
    t5_ab = data[(data['a_str'] == '11111') & (data['b_str'] == '00000')]
    t5_ba = data[(data['a_str'] == '00000') & (data['b_str'] == '11111')]
    
    agree_wadd = 0
    agree_total = 0
    if len(t5_ab) > 0:
        agree_wadd += (t5_ab['response'] == 0).sum()
        agree_total += len(t5_ab)
    if len(t5_ba) > 0:
        agree_wadd += (t5_ba['response'] == 1).sum()
        agree_total += len(t5_ba)
    p_agree = agree_wadd / agree_total if agree_total > 0 else 0.5
    
    # Conflict trials: WADD strongly favors '01111' but TTB favors '10000'
    t1_ab = data[(data['a_str'] == '01111') & (data['b_str'] == '10000')]
    t1_ba = data[(data['a_str'] == '10000') & (data['b_str'] == '01111')]
    
    conflict_wadd = 0
    conflict_total = 0
    if len(t1_ab) > 0:
        conflict_wadd += (t1_ab['response'] == 0).sum()
        conflict_total += len(t1_ab)
    if len(t1_ba) > 0:
        conflict_wadd += (t1_ba['response'] == 1).sum()
        conflict_total += len(t1_ba)
        
    p_conflict = conflict_wadd / conflict_total if conflict_total > 0 else 0.5
    
    return float(p_agree - p_conflict)
```

**Observed (real) value:** -0.0111 (var=0.0415)
**Predicted under pi_3:** 0.2022 (var=0.0418)
**Predicted under pi_4:** -0.0378 (var=0.0365)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed metric values hover consistently around 0.5 (or 0 difference in Experiment 6), indicating that subjects' choices are extremely noisy or heavily reliant on guessing. Theory 2 (Equal Weights with High Lapse) captures this baseline randomness quite well, especially in Experiments 1 and 6, by explicitly modeling a high guessing rate combined with a simple tallying heuristic. Theory 1 (Strategy Mixture of TTB and WADD) consistently overestimates the subjects' reliance on validities and structured strategies, predicting much higher alignment (e.g., 0.61 vs 0.50 in Exp 1, 0.62 vs 0.53 in Exp 2, and 0.20 vs -0.01 in Exp 6). However, Theory 2 also slightly misses the mark on Experiments 3 and 4 where a purely tallying approach introduces biases that aren't present in the data. Theory 1's structured approach is fundamentally mismatched to the near-random empirical data, making it the weaker theory overall.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory centered on 'Extreme Cognitive Noise / Single-Cue Focus'. Unlike Theory 2 which assumes tallying (equal weights), this theory should posit that subjects attempt to use the validities but are overwhelmed by cognitive friction, leading them to occasionally look only at the single highest-validity cue (a noisy TTB) while guessing randomly on the vast majority of trials. This differs from Theory 1 by completely dropping the complex Weighted Additive component and the structured mixture, and differs from Theory 2 by replacing the Tallying heuristic with a fragile, single-cue heuristic masked by an overwhelmingly high lapse rate. This will keep predictions tightly bounded near 0.5 while allowing for slight deviations based on the top validity rather than cue counts."
}
```

## Usage

```json
{
  "prompt_token_count": 6504,
  "candidates_token_count": 416,
  "total_token_count": 7785
}
```
