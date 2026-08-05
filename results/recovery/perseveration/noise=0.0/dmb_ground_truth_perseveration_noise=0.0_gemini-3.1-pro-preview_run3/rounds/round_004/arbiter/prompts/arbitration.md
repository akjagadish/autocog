# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_5" and "pi_6") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_5" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_6" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_5
Extreme Spatial Bias / Key Perseveration: Subjects completely disengage from the multi-attribute decision task and instead adopt a deterministic response strategy. They choose the exact same option (either always Option A or always Option B) on every single trial, completely ignoring the stimuli and feature validities. The preference for Option A versus Option B is fixed per subject, creating a population split between 'Always-A' and 'Always-B' responders. At the individual level, the choice policy is entirely deterministic and repetitive.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    # The subject adopts a deterministic response strategy, always choosing the same option.
    pref = int(parameters["preferred_option"])
    if pref == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## THEORY 2 — pi_6
First-Trial Inertia / Rational Disengagement: Subjects do not possess an innate spatial bias prior to the experiment. Instead, they actively evaluate the very first trial using a simple heuristic (like Tallying or Take-The-Best) to make an initial choice. However, due to the lack of trial-by-trial feedback and the repetitive nature of the multi-attribute stimuli, they rationally disengage after this first trial to minimize cognitive effort. For all subsequent trials, they exhibit 'first-trial inertia' by simply repeating their initial motor action. This produces 100% perseveration at the individual level, but grounds the origin of the bias in an initial stimulus-driven evaluation rather than an arbitrary trait.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, state, history):
    import numpy as np
    
    # If this is the first trial, evaluate using a simple heuristic
    if len(history.get("response", [])) == 0:
        state = np.asarray(state, dtype=float)
        a, b = state[0], state[1]
        
        heuristic_type = int(parameters["heuristic_type"])
        score_a, score_b = 0, 0
        
        if heuristic_type == 0:
            # Tallying (equal weights)
            score_a = np.sum(a)
            score_b = np.sum(b)
        else:
            # Take-The-Best (lexicographic)
            val = np.asarray(parameters["validities"], dtype=float)
            order = np.argsort(val)[::-1]
            for idx in order:
                if a[idx] > b[idx]:
                    score_a = 1
                    break
                elif b[idx] > a[idx]:
                    score_b = 1
                    break
                    
        if score_a > score_b:
            return np.array([1.0, 0.0])
        elif score_b > score_a:
            return np.array([0.0, 1.0])
        else:
            return np.array([0.5, 0.5])
            
    else:
        # Rational disengagement: repeat the initial motor action
        first_resp = history["response"][0]
        if first_resp == 0:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])

`policy(probs) -> int`:
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

## EXPERIMENT 1 (proposed by pi_5)

### DESIGN
**Validities (n_features=4):** [0.9, 0.8, 0.6, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]

**Rationale:** To dissociate the Advocated Theory (innate extreme spatial bias) from the Competing Theory (first-trial inertia based on heuristic evaluation), we exploit the randomized trial order. We include extreme dominance trials alongside compensatory trials that dissociate TTB from Tallying. The Competing Theory predicts that a subject's lifelong perseveration is perfectly predicted by the heuristic evaluation of their randomly assigned *first* trial (e.g., if the first trial is one where Option B dominates, they must become an 'Always-B' responder). The Advocated Theory predicts that the perseverated choice is an innate trait and thus entirely independent of the first trial's stimulus properties, allowing for 'Always-A' responders even when their very first trial objectively and strongly favored Option B.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



### METRIC
Rationale:
To achieve maximum discrimination, this metric exploits the pooled dataset to directly measure the core mechanism of the Competing Theory: deterministic heuristic evaluation of the first trial. If the Competing Theory is true, all subjects who randomly receive the exact same first trial will evaluate it using the same heuristic and make the identical choice, resulting in zero variance in responses conditioned on the first trial type. If the Advocated Theory is true, the choice is an innate spatial bias independent of the first trial, yielding maximum variance (~0.25) across subjects who saw the same first trial. The metric scales this conditional variance to produce a point estimate near 1.0 for the Competing Theory and 0.0 for the Advocated Theory. For the per-subject variance calculation, it falls back to measuring alignment with objectively dominant first trials, yielding a tight variance that guarantees an extremely high t-statistic.

Source:
def metric(data: pd.DataFrame) -> float:
    # Extract the chronologically first trial for each subject safely
    first_trials = data.drop_duplicates(subset=['subject_id'], keep='first').copy()
    
    if len(first_trials) > 1:
        # POOLED EVALUATION: Check if responses are deterministically driven by the first trial.
        # Create a string representation of the trial stimuli to group by.
        first_trials['trial_str'] = first_trials.apply(
            lambda row: ''.join(map(str, row['option_a_ratings'])) + ''.join(map(str, row['option_b_ratings'])), 
            axis=1
        )
        # Compute the variance of responses for each unique first trial.
        # Under Competing Theory, everyone with the same first trial uses the same heuristic and makes the same choice (var = 0).
        # Under Advocated Theory, innate biases are independent of the trial, so responses are mixed (var ~ 0.25).
        variances = first_trials.groupby('trial_str')['response'].var().fillna(0)
        
        # Scale so that 0 variance -> 1.0 (Competing), and 0.25 variance -> 0.0 (Advocated)
        return float(1.0 - 4.0 * variances.mean())
    else:
        # PER-SUBJECT EVALUATION (Fallback for variance calculation)
        # We measure alignment with objectively dominant trials to provide a compatible variance profile.
        row = first_trials.iloc[0]
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        r_signed = 1.0 if row['response'] == 1 else -1.0
        
        # Trial 4: Option B dominates
        if a == (0, 0, 0, 0) and b == (1, 1, 1, 1):
            return r_signed * 1.0
        # Trial 5: Option A dominates
        elif a == (1, 1, 1, 1) and b == (0, 0, 0, 0):
            return r_signed * -1.0
        else:
            return 0.0


### RESULTS
- Predicted under pi_5 (simulated): -0.0188 (var=0.9984)
- Predicted under pi_6 (simulated): 1.0000 (var=0.0000)
- Observed on real data: -0.0267 (var=0.9856)

## EXPERIMENT 2 (proposed by pi_6)

### DESIGN
**Validities (n_features=5):** [0.9, 0.7, 0.6, 0.6, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 2: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (First-Trial Inertia) from the Competing Theory (Extreme Spatial Bias), we exploit the randomized trial order. Both theories predict a subject will make the exact same choice (e.g., always A or always B) across all trials. However, the Advocated Theory predicts this perseveration is caused by a heuristic evaluation (Tallying or Take-The-Best) of the randomly assigned *first* trial. Thus, if a subject's first trial strongly favors Option A, they must become an 'Always-A' responder. The Competing Theory predicts perseveration is an innate trait, meaning a subject might be an 'Always-B' responder even if their very first trial strongly favored Option A. We include extreme dominance trials for A and B, as well as compensatory trials that pit the highest-validity cue against the sheer number of cues (dissociating TTB from Tallying). By analyzing the perseverated choice conditional on the first trial's stimulus properties, we can determine whether the bias is stimulus-driven or innate, and even identify which heuristic formed the initial evaluation.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



### METRIC
Rationale:
To optimally discriminate the theories, we map the subject's constant response to the stimulus properties of their very first trial. Under First-Trial Inertia (Advocated), subjects evaluating extreme dominance trials (T1 or T2) will always choose the objectively superior option, while their response on compensatory trials (T3 or T4) depends on their heuristic. Under Extreme Spatial Bias (Competing), responses are completely independent of the first trial. By assigning a score of +3 for choosing the dominant option on T1/T2, -5 for choosing the dominated option on T1/T2, and +1 for any response on T3/T4, we mathematically maximize the Welch's t-statistic. This specific scoring yields a high expected mean (2.0) and low variance (1.0) under the Advocated theory, and an expected mean of exactly 0.0 with higher variance (9.0) under the Competing theory, guaranteeing a highly significant p-value.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    scores = []
    for subj, df in data.groupby('subject_id', sort=False):
        first_row = df.iloc[0]
        a = first_row['option_a_ratings']
        b = first_row['option_b_ratings']
        sum_a = sum((int(x) for x in a))
        sum_b = sum((int(x) for x in b))
        resp = int(first_row['response'])
        if sum_a == 4 and sum_b == 1:
            score = 3.0 if resp == 0 else -5.0
        elif sum_a == 1 and sum_b == 4:
            score = -5.0 if resp == 0 else 3.0
        else:
            score = 1.0
        scores.append(score)
    if not scores:
        return 0.0
    return float(np.mean(scores))

### RESULTS
- Predicted under pi_5 (simulated): -1.1600 (var=15.9744)
- Predicted under pi_6 (simulated): 3.0000 (var=0.0000)
- Observed on real data: -0.8400 (var=15.9744)

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
**Predicted under pi_5:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.5000 (var=0.0000)

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
**Predicted under pi_5:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.5000 (var=0.0000)

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
**Predicted under pi_5:** 0.5000 (var=0.0000)
**Predicted under pi_6:** 0.5000 (var=0.0000)

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
**Predicted under pi_5:** 0.0000 (var=0.0000)
**Predicted under pi_6:** 0.0000 (var=0.0000)

### Experiment 7
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[0, 0, 0, 0]  B=[0, 0, 0, 0]
  A=[1, 1, 1, 1]  B=[1, 1, 1, 1]
  A=[1, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_5:** 0.0000 (var=0.0000)
**Predicted under pi_6:** 0.0000 (var=0.0000)

### Experiment 8
**Design**
  A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  A=[1, 0, 1, 0]  B=[1, 0, 1, 0]
  A=[0, 1, 0, 1]  B=[0, 1, 0, 1]

**Metric**
```python
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
```

**Observed (real) value:** 0.2500 (var=0.0000)
**Predicted under pi_5:** 0.2500 (var=0.0000)
**Predicted under pi_6:** 0.2500 (var=0.0000)

### Experiment 9
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    subject_means = data.groupby('subject_id')['response'].mean()
    majority_props = np.maximum(subject_means, 1.0 - subject_means)
    return float(majority_props.mean())
```

**Observed (real) value:** 1.0000 (var=0.0000)
**Predicted under pi_5:** 1.0000 (var=0.0000)
**Predicted under pi_6:** 1.0000 (var=0.0000)

### Experiment 10
**Design**
  A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  A=[0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Calculate the mean response (proportion of B choices) for each subject
    subj_means = data.groupby('subject_id')['response'].mean()
    # Find the proportion of the less frequent choice for each subject
    # Under the Competing Theory, subjects always choose A (0) or B (1), so this is exactly 0.0
    # Under the Advocated Theory, due to lapse rates and stimulus sensitivity, this will be > 0.0
    min_prop = subj_means.apply(lambda x: min(x, 1.0 - x))
    return float(min_prop.mean())
```

**Observed (real) value:** 0.0000 (var=0.0000)
**Predicted under pi_5:** 0.0000 (var=0.0000)
**Predicted under pi_6:** 0.0000 (var=0.0000)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Theory 1 (pi_5) accurately predicts the observed data across all experiments. Specifically, in Experiments 1 and 2, it correctly anticipates that responses are independent of the first trial's stimulus properties, yielding the observed high conditional variance. Theory 2 (pi_6) incorrectly assumes that subjects use a heuristic to evaluate the first trial before perseverating, which predicts zero variance conditional on the first trial's features\u2014a prediction heavily falsified by the real data. Since Theory 1 already perfectly captures the 100% perseveration rate (as shown by Exps 9 and 10) and its independence from stimuli (Exps 1 and 2), Theory 2 is degenerate and must be replaced.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a 'Random First Choice Inertia' theory. Instead of positing an innate, lifelong spatial trait (like Theory 1) or a heuristic evaluation of the first trial (like Theory 2), this theory should state that subjects simply guess randomly on the very first trial due to task ambiguity or low motivation. To minimize cognitive effort on all subsequent trials, they then deterministically repeat that initial random motor action. This provides a different psychological mechanism (random initial guess + effort-minimizing inertia) while successfully replicating the empirical reality of 100% perseveration that is entirely uncorrelated with the first trial's stimulus features."
}
```

## Usage

```json
{
  "prompt_token_count": 8725,
  "candidates_token_count": 317,
  "total_token_count": 10230
}
```
