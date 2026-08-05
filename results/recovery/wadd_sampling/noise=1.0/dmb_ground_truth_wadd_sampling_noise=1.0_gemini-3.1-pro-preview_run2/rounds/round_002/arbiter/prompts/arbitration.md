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
Decision-makers integrate all available information using a Weighted Additive (WADD) strategy, but scale evidence using the log-odds of cue validities. Rather than relying on a single best cue or ignoring cue validities, individuals compute an overall expected value for each option by taking the dot product of the option's features and the log-odds transformed validities. This transformation naturally stretches higher validities, balancing the influence of a single strong cue against multiple weaker ones. Choice is then executed stochastically via a softmax function, subject to occasional random lapses.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds to stretch higher validities
    # Clip to avoid division by zero or log(0) if validity is exactly 1.0
    val_clipped = np.clip(val, 0.0001, 0.9999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Calculate the weighted sum of features (expected value) for each option
    ev_a = np.dot(a, weights)
    ev_b = np.dot(b, weights)
    scores = np.array([ev_a, ev_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## THEORY 2 — pi_4
Cognitive Overload / Random Choice: When faced with multiple conflicting binary cues and explicitly stated validities, subjects experience cognitive overload or lack sufficient motivation to integrate the information systematically. Consequently, their decision-making process collapses into pure random guessing, treating both options as equally likely to be chosen regardless of their specific feature values.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    # The theory posits that cognitive overload causes subjects to guess randomly.
    # We extract the lapse_rate parameter to satisfy the parameter mapping requirement.
    lapse_rate = float(parameters["lapse_rate"])
    
    # Always return a uniform distribution over the two options.
    return np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.7, 0.6]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  trial 4: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 7: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 1]  B=[1, 0, 1, 0]

**Rationale:** This design aims to strongly dissociate the WADD (log-odds) theory from the Cognitive Overload / Random Choice theory. The Random Choice theory posits that subjects simply guess (50/50) on every trial regardless of the feature values. The WADD theory predicts systematic preferences based on the log-odds transformed validities. We include trials with overwhelming evidence for one option (where WADD predicts near-deterministic choice), trials where a single highly valid cue is pitted against multiple weaker cues, and trials with more balanced evidence. This spectrum of evidence strengths allows us to observe whether choice probabilities scale with the expected value difference (as WADD predicts) or remain flat at 0.5 (as Random Choice predicts).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of trials where the subject chooses the option with a higher total number of positive expert ratings, restricted to trials where the total number of positive ratings differs between the two options. The WADD model predicts that subjects will consistently choose the option with more positive ratings (since all cues have validities > 0.5), yielding a metric value close to 1.0. In contrast, the Cognitive Overload (Random Choice) model predicts that subjects will guess randomly, yielding a metric value of approximately 0.5.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    valid_trials = sum_a != sum_b
    if not valid_trials.any():
        return 0.5
        
    df_valid = data[valid_trials]
    sa = sum_a[valid_trials]
    sb = sum_b[valid_trials]
    
    chose_a = (df_valid['response'] == 0)
    chose_b = (df_valid['response'] == 1)
    
    correct = (chose_a & (sa > sb)) | (chose_b & (sb > sa))
    return float(correct.mean())

### RESULTS
- Predicted under pi_3 (simulated): 0.8550 (var=0.0107)
- Predicted under pi_4 (simulated): 0.5294 (var=0.0053)
- Observed on real data: 0.5167 (var=0.0071)

## EXPERIMENT 2 (proposed by pi_4)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 6: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 7: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Cognitive Overload (Random Choice) theory from the WADD (log-odds) theory, we must present trials across a wide spectrum of evidence strength. The Cognitive Overload theory predicts a flat 50/50 choice probability across all trials, regardless of the features presented. In contrast, the WADD theory predicts choice probabilities that scale with the difference in expected value (calculated using log-odds transformed validities). We include trials where one option completely dominates, trials where a single highly valid cue is pitted against multiple weaker cues, and trials with moderate evidence differences. If subjects are truly overloaded and guessing randomly, their choice distributions will remain at 0.5 even when one option overwhelmingly dominates the other (e.g., possessing all positive features).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



### METRIC
Rationale:
This metric calculates the proportion of times a subject chooses the strictly dominant option when faced with extreme trials (where one option has all positive features and the other has none). The Cognitive Overload theory predicts subjects will guess randomly even on these straightforward trials, yielding a metric value of roughly 0.5. The WADD theory predicts subjects will consistently choose the dominant option, yielding a metric value close to 1.0 (accounting for minor lapses).

Source:
def metric(data: pd.DataFrame) -> float:
    correct = 0
    total = 0
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        if a == (1, 1, 1, 1, 1) and b == (0, 0, 0, 0, 0):
            if row['response'] == 0:
                correct += 1
            total += 1
        elif a == (0, 0, 0, 0, 0) and b == (1, 1, 1, 1, 1):
            if row['response'] == 1:
                correct += 1
            total += 1
    return float(correct / total) if total > 0 else 0.5

### RESULTS
- Predicted under pi_3 (simulated): 0.8425 (var=0.0095)
- Predicted under pi_4 (simulated): 0.4875 (var=0.0094)
- Observed on real data: 0.5183 (var=0.0092)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    matches = 0
    total = 0
    for a, b, r in zip(data['option_a_ratings'], data['option_b_ratings'], data['response']):
        for i in range(len(a)):
            if a[i] > b[i]:
                if r == 0:
                    matches += 1
                total += 1
                break
            elif b[i] > a[i]:
                if r == 1:
                    matches += 1
                total += 1
                break
    return float(matches / total) if total > 0 else 0.5
```

**Observed (real) value:** 0.4900 (var=0.0012)
**Predicted under pi_3:** 0.5325 (var=0.0022)
**Predicted under pi_4:** 0.4977 (var=0.0023)

### Experiment 4
**Design**
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 0]
  A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  A=[0, 1, 1, 0]  B=[1, 0, 0, 1]
  A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  A=[0, 1, 1, 1]  B=[1, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    match_tallying = 0
    total_strict = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        tally_a = np.sum(a > b)
        tally_b = np.sum(b > a)
        if tally_a > tally_b:
            tally_pred = 0
        elif tally_b > tally_a:
            tally_pred = 1
        else:
            continue
        total_strict += 1
        if row['response'] == tally_pred:
            match_tallying += 1
    return match_tallying / total_strict if total_strict > 0 else 0.5
```

**Observed (real) value:** 0.5020 (var=0.0043)
**Predicted under pi_3:** 0.3943 (var=0.0036)
**Predicted under pi_4:** 0.4993 (var=0.0037)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  A=[0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    def is_target_trial(row):
        a = tuple(row['option_a_ratings'])
        # These are trials 1, 3, 4, and 5 from the design.
        # In these trials, WADD strongly prefers Option A due to log-odds weighting of highly valid cues,
        # whereas Tallying strictly prefers Option B because B has more winning features.
        return a in [
            (1, 0, 0, 0, 0),
            (1, 0, 1, 0, 0),
            (0, 1, 0, 0, 0),
            (1, 0, 0, 1, 0)
        ]
        
    mask = data.apply(is_target_trial, axis=1)
    target_data = data[mask]
    if len(target_data) == 0:
        return 0.5
        
    # Return the proportion of times Option A was chosen (response == 0)
    return float((target_data['response'] == 0).mean())
```

**Observed (real) value:** 0.5125 (var=0.0033)
**Predicted under pi_3:** 0.7296 (var=0.0132)
**Predicted under pi_4:** 0.4954 (var=0.0043)

### Experiment 6
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 1]  B=[1, 0, 0, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the number of winning features for each option
    a_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if a > b), axis=1)
    b_wins = data.apply(lambda row: sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if b > a), axis=1)
    
    # Filter out trials where Tallying predicts a tie
    mask = a_wins != b_wins
    filtered_data = data[mask]
    if len(filtered_data) == 0:
        return 0.5
        
    a_wins_f = a_wins[mask]
    b_wins_f = b_wins[mask]
    
    # Determine the Tallying prediction (0 for A, 1 for B)
    tally_prediction = (b_wins_f > a_wins_f).astype(int)
    
    # Return the proportion of choices that align with the Tallying prediction
    return (filtered_data['response'] == tally_prediction).mean()
```

**Observed (real) value:** 0.5067 (var=0.0022)
**Predicted under pi_3:** 0.4448 (var=0.0025)
**Predicted under pi_4:** 0.5055 (var=0.0021)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across all six experiments, the observed metric values hover consistently around 0.50. This indicates that subjects are either guessing randomly or employing strategies that, in aggregate, are indistinguishable from random chance on these metrics. THEORY 2 (Cognitive Overload / Random Choice) predicts metric values near 0.50 for all experiments and therefore matches the observed data extremely well. THEORY 1 (WADD with log-odds validities) predicts strong systematic preferences (e.g., ~0.85 in Exps 1 and 2, ~0.73 in Exp 5) that are completely absent in the real data. Since THEORY 2 is effectively a null model that accurately captures the lack of systematicity, THEORY 1 is degenerate and should be replaced with a stronger, classic heuristic competitor to ensure that no systematic bounded-rationality strategy is being overlooked.",
  "verdict": "new_theory",
  "target_theory_idx": 1,
  "recommendation": "Propose the 'Take-The-Best' (TTB) heuristic as a brand-new theory. TTB posits that decision-makers do not integrate all information (unlike WADD) nor do they guess completely randomly (unlike Cognitive Overload). Instead, they search through cues in descending order of their explicitly stated validities and base their choice entirely on the first cue that discriminates between the two options. Implementing TTB with a lapse rate will provide a strong, well-established bounded rationality baseline to test whether there is any residual systematicity in the data that the Random Choice model might be missing."
}
```

## Usage

```json
{
  "prompt_token_count": 5924,
  "candidates_token_count": 341,
  "total_token_count": 7272
}
```
