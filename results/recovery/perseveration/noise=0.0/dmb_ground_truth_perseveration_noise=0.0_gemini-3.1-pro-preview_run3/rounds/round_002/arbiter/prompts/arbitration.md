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
Subjects exhibit complete disengagement from the multi-attribute decision task. Rather than evaluating the options based on their features and the validities of those features, subjects ignore all stimulus information and simply guess at random on every trial, choosing Option A or Option B with equal probability.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The model completely ignores the state and history, 
    # reflecting total task disengagement.
    return np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_3
Decision-makers use a Weighted Additive (WADD) strategy, integrating all available features weighted by their validities, but they are subject to significant spatial/positional biases (e.g., a baseline preference for Option A over Option B) and high levels of task disengagement (lapse rate). In environments where subjects ignore features, the positional bias and lapse rate dominate the choice, leading to choices that appear completely orthogonal to standard heuristic predictions like TTB or Tallying (yielding exact 0.5 consistency).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Weighted Additive (WADD) scores
    score_a = np.sum(val * a)
    score_b = np.sum(val * b)
    
    beta = float(parameters["beta"])
    bias_a = float(parameters["bias_A"])
    epsilon = float(parameters["epsilon"])
    
    # Incorporate spatial/positional bias for Option A
    logits = np.array([beta * score_a + bias_a, beta * score_b])
    
    # Numerically stable softmax
    logits = logits - np.max(logits)
    p_core = np.exp(logits) / np.sum(np.exp(logits))
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])

`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))

## EXPERIMENT 1 (proposed by pi_4)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 7: A=[0, 0, 0, 0]  B=[0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1]  B=[1, 1, 1, 1]
  trial 9: A=[1, 0, 1, 0]  B=[1, 0, 1, 0]
  trial 10: A=[0, 1, 0, 1]  B=[0, 1, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (pure random guessing / complete disengagement) from the Competing Theory (WADD with spatial bias and lapse rate), this design includes three types of trials: (1) Extreme dominance trials where Option A or Option B has all positive features, maximizing the WADD difference. (2) Compensatory trials where the highest-validity cue opposes the sum of lower-validity cues, testing sensitivity to weighted feature integration. (3) Identical-profile (tie) trials where both options have the exact same features. The Advocated Theory rigidly predicts exactly 50/50 choice probabilities across all trial types. In contrast, the Competing Theory will predict strong deviations from 50/50 on dominance and compensatory trials (via the beta parameter) and will reveal any baseline positional preference on the identical-profile trials (via the bias_A parameter).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Under the Advocated Theory (pure random guessing), subjects are entirely insensitive to the features, so the probability of choosing Option A should be approximately 0.5 regardless of whether Option A has more positive features than Option B. The difference in the choice rate of A between these two conditions will be 0. Under the Competing Theory, subjects use a Weighted Additive strategy, meaning they are more likely to choose Option A when it has more positive features than Option B. This metric calculates the difference in the rate of choosing Option A when it 'dominates' (has more positive features) versus when Option B 'dominates', robustly capturing sensitivity to feature values.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate sum of ratings for A and B
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Identify trials where A has more positive features than B, and vice versa
    a_dom = (sum_a > sum_b)
    b_dom = (sum_a < sum_b)
    
    # Proportion of choosing A (response == 0)
    p_a_given_a_dom = (data.loc[a_dom, 'response'] == 0).mean()
    p_a_given_b_dom = (data.loc[b_dom, 'response'] == 0).mean()
    
    if np.isnan(p_a_given_a_dom):
        p_a_given_a_dom = 0.5
    if np.isnan(p_a_given_b_dom):
        p_a_given_b_dom = 0.5
        
    return float(p_a_given_a_dom - p_a_given_b_dom)

### RESULTS
- Predicted under pi_4 (simulated): -0.0600 (var=0.0318)
- Predicted under pi_3 (simulated): 0.1700 (var=0.1160)
- Observed on real data: 0.0000 (var=0.0000)

## EXPERIMENT 2 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 7: A=[1, 0, 1, 0]  B=[1, 0, 1, 0]
  trial 8: A=[0, 1, 0, 1]  B=[0, 1, 0, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (WADD with spatial bias and lapse rate) from the Competing Theory (pure random guessing), we must expose variations in choice probabilities that random guessing cannot capture. The Competing Theory rigidly predicts a 50/50 choice split on every trial, regardless of the stimuli. The Advocated Theory predicts that choices will track the weighted sums of features (modulated by sensitivity 'beta') and may exhibit a baseline preference for Option A (via 'bias_A'). We include extreme dominance trials (where one option has all positive features and the other has none) to elicit maximum deviation from 50/50. We also include compensatory trials to test sensitivity to feature integration, and identical-profile (tie) trials where both options have the same features. In tie trials, random guessing still predicts exactly 0.5, whereas the Advocated Theory will reveal any underlying spatial bias.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
The Competing Theory predicts random guessing, meaning subjects should choose Option A roughly 50% of the time on every trial type. For 12 repetitions per trial type, the squared deviation of the choice proportion from 0.5 will tightly cluster around its expected value (approx. 0.021). In contrast, the Advocated Theory posits that subjects use a Weighted Additive strategy and exhibit a spatial bias. Depending on their spatial bias and sensitivity (beta) to the features, their choice probabilities will systematically diverge from 0.5 on most trial types (often approaching 0 or 1). Thus, the mean squared deviation from 0.5 across trial types will be significantly and reliably higher for the Advocated Theory.

Source:
def metric(data: pd.DataFrame) -> float:
    df = data.copy()
    # Convert responses to 1 if Option A was chosen, 0 otherwise
    df['is_A'] = (df['response'] == 0).astype(float)
    
    # Create a hashable string representation for each unique trial type
    df['trial_type'] = df['option_a_ratings'].apply(lambda x: ''.join(map(str, x))) + "_" + \
                       df['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # Calculate the proportion of times Option A was chosen for each trial type, per subject
    subj_trial_means = df.groupby(['subject_id', 'trial_type'])['is_A'].mean()
    
    # Calculate the squared deviation from 0.5 (random guessing baseline)
    sq_dev = (subj_trial_means - 0.5) ** 2
    
    # Average these squared deviations across trial types for each subject
    subj_scores = sq_dev.groupby('subject_id').mean()
    
    # Return the mean across all subjects
    return float(subj_scores.mean())

### RESULTS
- Predicted under pi_4 (simulated): 0.0186 (var=0.0001)
- Predicted under pi_3 (simulated): 0.0813 (var=0.0035)
- Observed on real data: 0.2500 (var=0.0000)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_preds = []
    for a, b in zip(data['option_a_ratings'], data['option_b_ratings']):
        pred = 0.5
        for i in range(len(a)):
            if a[i] > b[i]:
                pred = 0
                break
            elif b[i] > a[i]:
                pred = 1
                break
        ttb_preds.append(pred)
        
    ttb_preds = np.array(ttb_preds)
    responses = data['response'].values
    
    return float(np.mean(ttb_preds == responses))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.5133 (var=0.0022)
**Predicted under pi_3:** 0.4029 (var=0.0166)

### Experiment 4
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Extract option ratings into 2D numpy arrays
    a_ratings = np.array(data['option_a_ratings'].tolist())
    b_ratings = np.array(data['option_b_ratings'].tolist())
    
    # Calculate number of feature-wise wins for each option
    a_wins = np.sum(a_ratings > b_ratings, axis=1)
    b_wins = np.sum(b_ratings > a_ratings, axis=1)
    
    responses = data['response'].values
    
    # Identify trials where Tallying has a strict preference
    tally_prefers_a = a_wins > b_wins
    tally_prefers_b = b_wins > a_wins
    
    # Check if subject's response is consistent with Tallying's preference
    consistent = (tally_prefers_a & (responses == 0)) | (tally_prefers_b & (responses == 1))
    strict_trials = tally_prefers_a | tally_prefers_b
    
    if np.sum(strict_trials) == 0:
        return 0.5
        
    # Return the proportion of Tallying-consistent choices on strict trials
    return float(np.sum(consistent[strict_trials]) / np.sum(strict_trials))
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.5134 (var=0.0042)
**Predicted under pi_3:** 0.5791 (var=0.0214)

### Experiment 5
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 0, 1, 1]
  A=[1, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[0, 1, 0, 0]
  A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  A=[0, 0, 0, 1]  B=[0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.95, 0.75, 0.65, 0.55])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # TTB predictions
    diff = a_mat - b_mat
    ttb_preds = np.zeros(len(data))
    for i in range(len(data)):
        for j in range(4):
            if diff[i, j] > 0:
                ttb_preds[i] = 0
                break
            elif diff[i, j] < 0:
                ttb_preds[i] = 1
                break
                
    # WADD predictions (without spatial bias)
    score_a = a_mat @ validities
    score_b = b_mat @ validities
    wadd_preds = (score_b > score_a).astype(int)
    
    # Isolate trials where TTB and WADD (unbiased) predict opposite choices
    mask = ttb_preds != wadd_preds
    
    if not np.any(mask):
        return 0.5
        
    responses = data['response'].values
    agreement = (responses[mask] == ttb_preds[mask]).mean()
    return float(agreement)
```

**Observed (real) value:** 0.5000 (var=0.0000)
**Predicted under pi_4:** 0.4997 (var=0.0040)
**Predicted under pi_3:** 0.4614 (var=0.0072)

### Experiment 6
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[1, 1, 0, 0]  B=[1, 1, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    def is_t1(row):
        return tuple(int(x) for x in row['option_a_ratings']) == (0, 1, 1, 1) and \
               tuple(int(x) for x in row['option_b_ratings']) == (1, 0, 0, 0)
               
    def is_t2(row):
        return tuple(int(x) for x in row['option_a_ratings']) == (1, 0, 0, 0) and \
               tuple(int(x) for x in row['option_b_ratings']) == (0, 1, 1, 1)
               
    t1_mask = data.apply(is_t1, axis=1)
    t2_mask = data.apply(is_t2, axis=1)
    
    t1_data = data[t1_mask]
    t2_data = data[t2_mask]
    
    p_a_t1 = (t1_data['response'] == 0).mean() if len(t1_data) > 0 else 0.5
    p_a_t2 = (t2_data['response'] == 0).mean() if len(t2_data) > 0 else 0.5
    
    return float(p_a_t1 - p_a_t2)
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_4:** -0.0050 (var=0.0484)
**Predicted under pi_3:** 0.1400 (var=0.0701)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all experiments, the observed data reveals a striking pattern: the difference in choice proportions between any two trial types is always exactly 0.0000 (as seen in Exp 1 and Exp 6), and the agreement with feature-based strategies like TTB or Tallying is exactly 0.5000 (Exp 3, 4, 5). Most importantly, in Experiment 2, the mean squared deviation from a 0.5 choice proportion is exactly 0.2500 with a between-subject variance of 0. A squared deviation of 0.25 from 0.5 means the choice proportions are either exactly 1.0 or 0.0 for every trial type. This implies that subjects are completely ignoring the stimulus features and deterministically choosing the exact same option (e.g., always Option A or always Option B) on every single trial. Theory 1 predicts random guessing (which would yield a squared deviation close to 0), and Theory 2 predicts a mix of WADD and bias. Neither captures this deterministic key-press perseveration.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose a new theory of 'Extreme Spatial Bias / Key Perseveration'. The theory should state that subjects completely disengage from the multi-attribute decision task and instead adopt a deterministic response strategy, choosing the exact same option (either always Option A or always Option B) on every single trial regardless of the stimuli. The model should have a parameter dictating the probability of a subject being an 'Always-A' versus 'Always-B' responder, but at the individual level, the policy should be deterministic."
}
```

## Usage

```json
{
  "prompt_token_count": 6260,
  "candidates_token_count": 373,
  "total_token_count": 7934
}
```
