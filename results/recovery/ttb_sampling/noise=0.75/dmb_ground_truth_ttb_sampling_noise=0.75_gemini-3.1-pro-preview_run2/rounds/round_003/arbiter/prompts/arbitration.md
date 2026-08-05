# arbitration

## System Prompt

You are a renowned cognitive scientist arbitrating between two theories across multiple experiments in the Decision Making (Binary Features) domain.

Each experiment was proposed alongside a metric and an expected outcome. For each experiment you are shown the design, the metric, both theories' predicted metric values (from simulated data), and the observed metric value on real data. The two theories are tagged by stable labels (e.g. "pi_3" and "pi_5") and the same labels are reused on each experiment's predictions.

Each metric value below is shown as `point_estimate (var=X)`, where `point_estimate` is `metric(data)` evaluated on the full pooled dataset and `var` is the population (between-subject) variance of the same metric re-applied per `subject_id`. The point estimate is the canonical scalar; `var` reports how consistent that estimate is across subjects (lower = more consistent). `var=n/a` means the metric could not be applied to a single-subject slice.

The goal of the arbitration is to surface theories that are task-invariant: that is, theories that can explain data across all experiments in the same domain. Perform a deep dive: which among the two theories better captures the observed data?Do not just look at the newest experiments, but look at all experiments together. If a theory is good at explaining all the data keep it. However, if the both theories are good at explaining some experiment but not all, then it might be a good idea to propose a completly new theory that can potentially explain all the data. It is often better to propose a new theory than to propose a new model. Even if one theory is clearly better than the other, instead of proposing a new model, you can propose a new theory that is a stronger competitor to the winning theory instead of proposing a new model. Only propose a new model if both theories are very good and you are confident that the new model will be better than the current one clearly distinguish the two theories.  Then issue a verdict: either "new_model" (keep the current theory description, but regenerate new predict / policy / parameter ranges such that they better capture the observed data across all experiments) or  "new_theory" (the current theory is degenerate; propose a brand-new theory that can better capture the observed data across all experiments). `target_theory_idx` is 1 if you are acting on the theory labelled "pi_3" (THEORY 1 below), or 2 if you are acting on the theory labelled "pi_5" (THEORY 2 below). Justify your choice.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

## THEORY 1 — pi_3
Take-The-Best (TTB) heuristic: People make decisions by ranking features according to their validities and choosing the option that is favored by the single most valid discriminating feature. If no feature discriminates, they guess. This is a lexicographic, non-compensatory strategy. However, human execution of this strategy is highly noisy, so choice probabilities are heavily tempered by response noise (low beta) and random guessing lapses (high epsilon).

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Rank features by validity in descending order.
    # We use a stable sort to preserve the original feature order in case of ties.
    ranked_features = np.argsort(validities, kind='stable')[::-1]
    
    scores = np.array([0.0, 0.0])
    # Find the first feature that discriminates between the two options
    for idx in ranked_features:
        if a[idx] > b[idx]:
            scores[0] = 1.0
            break
        elif b[idx] > a[idx]:
            scores[1] = 1.0
            break
            
    # If no feature discriminates, the core preference is uniform
    if scores[0] == 0.0 and scores[1] == 0.0:
        p_core = np.array([0.5, 0.5])
    else:
        beta = float(parameters["beta"])
        # Softmax over the scores to introduce response noise
        z = beta * scores
        z = z - np.max(z)
        e = np.exp(z)
        p_core = e / np.sum(e)
        
    epsilon = float(parameters["epsilon"])
    n_opts = p_core.shape[0]
    
    # Blend with uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)


## THEORY 2 — pi_5
Probabilistic Cue Selection (Stochastic TTB): Human decision-makers use a non-compensatory, one-reason heuristic but are stochastic in their cue retrieval. Instead of strictly ranking features by validity, subjects sample a feature to evaluate with a probability proportional to its validity (via a softmax). If the sampled feature discriminates between the options, they base their choice entirely on that feature. If it does not discriminate, they sample again. This maintains the non-compensatory nature of the decision while naturally introducing variability in which cue is selected, offering a mechanistic explanation for choice noise without relying on compensatory tallying.

`predict(parameters, state, history) -> np.ndarray`:
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")

    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify which features discriminate between A and B
    discrim = (a != b)
    
    if not np.any(discrim):
        # If no features discriminate, the decision is a random guess
        p_core = np.array([0.5, 0.5])
    else:
        # The subject repeatedly samples features proportional to exp(gamma * validity)
        # until a discriminating feature is found. The probability that the first
        # discriminating feature found is feature i is equivalent to a softmax over
        # the validities restricted to the set of discriminating features.
        z = gamma * validities[discrim]
        z = z - np.max(z)  # For numerical stability
        w = np.exp(z)
        w = w / np.sum(w)
        
        # The chosen discriminating feature dictates the choice entirely.
        # Sum the probabilities of sampling a feature that favors A vs B.
        favor_a = (a[discrim] > b[discrim])
        favor_b = (b[discrim] > a[discrim])
        
        p_a = np.sum(w[favor_a])
        p_b = np.sum(w[favor_b])
        
        p_core = np.array([p_a, p_b])
        
    # Blend with a uniform lapse rate for general response noise/inattention
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])


`policy(probs) -> int`:
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)


## EXPERIMENT 1 (proposed by pi_3)

### DESIGN
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 6: A=[1, 1, 1, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 0, 1, 1]  B=[1, 0, 1, 1, 1]
  trial 8: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 1]
  trial 9: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 10: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]

**Rationale:** To quantitatively dissociate Take-The-Best (TTB) from Probabilistic Cue Selection (Stochastic TTB), we manipulate the number and validities of subordinate discriminating features while holding the single best discriminating feature constant. Under pure TTB, the decision is strictly based on the most valid discriminating feature; the presence, number, or validities of other lower-ranking discriminating features have no effect on the predicted choice probabilities (the pre-noise score is always 1 for the TTB winner and 0 for the loser). In contrast, Stochastic TTB samples among all discriminating features with probabilities proportional to exp(gamma * validity). Thus, Stochastic TTB predicts that the choice probability for the option favored by the best cue will decrease as more subordinate cues favor the opposing option, whereas pure TTB predicts a constant choice probability across these conditions. By systematically varying the opposing subordinate cues, we can directly test whether choice probabilities are invariant to subordinate cues (TTB) or sensitive to them (Stochastic TTB).

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



### METRIC
Rationale:
Under Probabilistic Cue Selection (STTB), the decision-maker samples a discriminating feature proportional to its validity. On trials where exactly one feature discriminates between the options (Hamming distance of 1), STTB will sample that single feature with probability 1.0, yielding a pre-lapse choice probability of exactly 1.0 for the favored option, regardless of the gamma parameter. In contrast, under the strict Take-The-Best (TTB) model, the pre-lapse choice probability is determined by a softmax over the feature scores, tempered by the beta parameter. Since beta is bounded in the parameter space (max 2.0), TTB's pre-lapse probability is strictly bounded (max ~0.88). Because both models are subject to the same epsilon lapse rates, the absolute observed choice rate for the favored option on these single-discriminating-feature trials will be substantially and reliably higher for STTB than for TTB, providing a clean separation between the theories.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert lists of binary ratings into 2D numpy arrays
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Find trials with exactly 1 discriminating feature (Hamming distance == 1)
    hamming_dist = np.sum(a_ratings != b_ratings, axis=1)
    mask = hamming_dist == 1
    
    if not np.any(mask):
        return 0.5
        
    # Determine the favored option on these trials.
    # Since they differ by exactly one feature, the option with the '1' 
    # on that feature will have a strictly greater sum of ratings.
    a_sum = np.sum(a_ratings[mask], axis=1)
    b_sum = np.sum(b_ratings[mask], axis=1)
    
    # favored_option is 0 (Option A) if A has the higher sum, else 1 (Option B)
    favored_option = (b_sum > a_sum).astype(int)
    responses = data.loc[mask, 'response'].values
    
    # Return the absolute choice rate of the favored option on these specific trials
    return float(np.mean(responses == favored_option))

### RESULTS
- Predicted under pi_3 (simulated): 0.6011 (var=0.0181)
- Predicted under pi_5 (simulated): 0.7711 (var=0.0251)
- Observed on real data: 0.5844 (var=0.0223)

## EXPERIMENT 2 (proposed by pi_5)

### DESIGN
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=14):**
  trial 1: A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 1, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 3: A=[1, 1, 1, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[0, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 6: A=[0, 1, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 7: A=[0, 1, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 8: A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 9: A=[1, 1, 1, 1, 0]  B=[1, 0, 1, 1, 1]
  trial 10: A=[1, 1, 1, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 11: A=[1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 12: A=[1, 0, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 13: A=[1, 0, 1, 0, 1]  B=[1, 1, 1, 1, 0]
  trial 14: A=[1, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Rationale:** To quantitatively dissociate Probabilistic Cue Selection (Stochastic TTB) from standard Take-The-Best (TTB), this design manipulates the number of *supporting* subordinate features while holding the single best discriminating feature and the opposing feature constant. Standard TTB's predicted choice probability is driven entirely by the single most valid discriminating feature; adding more lower-validity features that support the same option does not change the TTB score (it remains 1 for the winner and 0 for the loser), thus predicting a perfectly flat choice probability across these conditions. In contrast, Stochastic TTB samples among all discriminating features. By adding more features that support the TTB winner, the total probability mass for sampling a feature favoring that option increases. Thus, Stochastic TTB predicts a graded increase in choice probability as supporting features are added, whereas standard TTB predicts no change.

**Computed schedule:** 14 unique pairs × 6 reps = 84 trials per subject.



### METRIC
Rationale:
To robustly discriminate Stochastic TTB from standard TTB, we must isolate the effect of subordinate cues while controlling for the massive between-subject variance caused by noise parameters (epsilon and beta). Standard TTB bases its decision exclusively on the single highest-validity discriminating feature. Consequently, its true choice probability for the favored option is mathematically identical across all trial types, regardless of how many subordinate cues support or oppose it. Thus, the expected determinism (measured by the squared deviation of the empirical choice proportion from 0.5) is identical for all trial types under TTB. In contrast, Stochastic TTB samples among all discriminating features. When multiple subordinate features support the best feature ('Strong' trials, n_support >= 3), the probability mass heavily favors that option, leading to highly deterministic choices. When subordinate features oppose the best feature ('Weak' trials, n_support == 1), the probability mass is diluted, pulling the choice probability closer to 0.5 and reducing determinism. By computing the within-subject difference in determinism (p_j - 0.5)^2 between Strong and Weak trials, we establish a metric where standard TTB predicts exactly 0.0 with extremely low variance (since the baseline determinism cancels out). Stochastic TTB, however, predicts a strictly positive difference. This within-subject control perfectly isolates the mechanistic difference between the theories while neutralizing noise parameter confounds.

Source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd

    def subject_metric(sub_df):
        a_ratings = np.array(sub_df['option_a_ratings'].tolist())
        b_ratings = np.array(sub_df['option_b_ratings'].tolist())
        diff = a_ratings - b_ratings
        
        n_trials = len(diff)
        n_support = np.zeros(n_trials, dtype=int)
        for i in range(n_trials):
            d = diff[i]
            nz = np.where(d != 0)[0]
            if len(nz) > 0:
                fd = d[nz[0]]
                n_support[i] = np.sum((d * fd) > 0)
        
        trial_keys = sub_df['option_a_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x])) + '_' + \
                     sub_df['option_b_ratings'].apply(lambda x: ''.join([str(int(v)) for v in x]))
        
        df_temp = pd.DataFrame({
            'trial_key': trial_keys,
            'response': sub_df['response'].values,
            'n_support': n_support
        })
        
        grouped = df_temp.groupby('trial_key')
        p_j = grouped['response'].mean()
        n_supp_j = grouped['n_support'].first()
        
        # D_j measures the determinism of the choices for trial type j
        D_j = (p_j - 0.5) ** 2
        
        strong_mask = n_supp_j >= 3
        weak_mask = n_supp_j == 1
        
        if np.sum(strong_mask) == 0 or np.sum(weak_mask) == 0:
            return 0.0
            
        # Return the difference in determinism between Strong and Weak support trials
        return float(D_j[strong_mask].mean() - D_j[weak_mask].mean())

    if 'subject_id' in data.columns and data['subject_id'].nunique() > 1:
        return float(data.groupby('subject_id').apply(subject_metric).mean())
    else:
        return float(subject_metric(data))

### RESULTS
- Predicted under pi_3 (simulated): -0.0024 (var=0.0019)
- Predicted under pi_5 (simulated): 0.0056 (var=0.0019)
- Observed on real data: -0.0080 (var=0.0019)

## PERFORMANCE FOR THESE TWO THEORIES ON OTHER EXPERIMENTS
### Experiment 3
**Design**
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]

**Metric**
```python
import numpy as np
import pandas as pd

def metric(data: pd.DataFrame) -> float:
    tally_choices = []
    for _, row in data.iterrows():
        a = tuple(row['option_a_ratings'])
        b = tuple(row['option_b_ratings'])
        
        # Trial 1: A has 3 wins on weak features, B has 2 wins on strong features
        if a == (0, 0, 1, 1, 1) and b == (1, 1, 0, 0, 0):
            # Tallying prefers A
            tally_choices.append(1.0 if row['response'] == 0 else 0.0)
        # Trial 2: A has 2 wins on strong features, B has 3 wins on weak features
        elif a == (1, 1, 0, 0, 0) and b == (0, 0, 1, 1, 1):
            # Tallying prefers B
            tally_choices.append(1.0 if row['response'] == 1 else 0.0)
            
    if not tally_choices:
        return 0.5
    return float(np.mean(tally_choices))
```

**Observed (real) value:** 0.3975 (var=0.0118)
**Predicted under pi_3:** 0.3862 (var=0.0217)
**Predicted under pi_5:** 0.3100 (var=0.0406)

### Experiment 4
**Design**
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    # Identify trials where the options are [1, 1, 0, 0, 0] and [0, 0, 1, 1, 1]
    is_11000_A = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    is_11000_B = data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    mask = is_11000_A | is_11000_B
    if mask.sum() == 0:
        return 0.5
    
    subset = data[mask]
    is_11000_A_sub = is_11000_A[mask]
    
    # Check if the subject chose the option with [1, 1, 0, 0, 0]
    chose_11000 = (is_11000_A_sub & (subset['response'] == 0)) | (~is_11000_A_sub & (subset['response'] == 1))
    return float(chose_11000.mean())
```

**Observed (real) value:** 0.5833 (var=0.0110)
**Predicted under pi_3:** 0.5883 (var=0.0171)
**Predicted under pi_5:** 0.7042 (var=0.0232)

### Experiment 5
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    ttb_match_count = 0
    total_incongruent = 0
    
    validities = [0.95, 0.85, 0.8, 0.75, 0.7]
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        resp = row['response']
        
        ttb_choice = None
        for i in range(5):
            if a[i] > b[i]:
                ttb_choice = 0
                break
            elif b[i] > a[i]:
                ttb_choice = 1
                break
        
        wadd_score_a = sum(a[i] * v for i, v in enumerate(validities))
        wadd_score_b = sum(b[i] * v for i, v in enumerate(validities))
        wadd_choice = 0 if wadd_score_a > wadd_score_b else 1
        
        if ttb_choice is not None and ttb_choice != wadd_choice:
            total_incongruent += 1
            if resp == ttb_choice:
                ttb_match_count += 1
                
    if total_incongruent == 0:
        return 0.5
    return ttb_match_count / total_incongruent

```

**Observed (real) value:** 0.5876 (var=0.0025)
**Predicted under pi_3:** 0.6240 (var=0.0143)
**Predicted under pi_5:** 0.6467 (var=0.0198)

### Experiment 6
**Design**
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.vstack(data['option_a_ratings'].values)
    b_ratings = np.vstack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    diff = a_ratings - b_ratings
    
    ttb_choices = []
    for i in range(len(diff)):
        choice = -1
        for j in range(diff.shape[1]):
            if diff[i, j] > 0:
                choice = 0
                break
            elif diff[i, j] < 0:
                choice = 1
                break
        ttb_choices.append(choice)
        
    ttb_choices = np.array(ttb_choices)
    valid = ttb_choices != -1
    
    if not np.any(valid):
        return 0.5
        
    return float(np.mean(responses[valid] == ttb_choices[valid]))
```

**Observed (real) value:** 0.5893 (var=0.0033)
**Predicted under pi_3:** 0.6031 (var=0.0082)
**Predicted under pi_5:** 0.6962 (var=0.0181)

### Experiment 7
**Design**
  A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  A=[1, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]
  A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  A=[0, 0, 1, 1, 0]  B=[1, 1, 0, 0, 1]
  A=[0, 0, 0, 0, 1]  B=[1, 1, 1, 1, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    high_support_trials = []
    low_support_trials = []
    
    for _, row in data.iterrows():
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        
        # Feature 0 has the highest validity in this design
        a0 = a[0]
        b0 = b[0]
        
        if a0 > b0:
            ttb_winner = 0
            ttb_support = sum(a)
        elif b0 > a0:
            ttb_winner = 1
            ttb_support = sum(b)
        else:
            continue
            
        chose_ttb = 1 if row['response'] == ttb_winner else 0
        
        if ttb_support == 4:
            high_support_trials.append(chose_ttb)
        elif ttb_support == 1:
            low_support_trials.append(chose_ttb)
            
    m_high = np.mean(high_support_trials) if len(high_support_trials) > 0 else 0.5
    m_low = np.mean(low_support_trials) if len(low_support_trials) > 0 else 0.5
    
    return float(m_high - m_low)
```

**Observed (real) value:** -0.0133 (var=0.0336)
**Predicted under pi_3:** -0.0100 (var=0.0295)
**Predicted under pi_5:** 0.0644 (var=0.0304)

### Experiment 8
**Design**
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Metric**
```python
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Determine which option is the Take-The-Best (TTB) winner.
    # Feature 0 has the highest validity (0.95), so the option with a 1 in feature 0 is the TTB winner.
    ttb_winner = data['option_a_ratings'].apply(lambda x: 0 if x[0] == 1 else 1)
    chose_ttb = (data['response'] == ttb_winner).astype(float)
    
    # Calculate how many features the opposing (TTB-losing) option has.
    def get_opposing_cues(row):
        if row['option_a_ratings'][0] == 1:
            return sum(row['option_b_ratings'])
        else:
            return sum(row['option_a_ratings'])
            
    opposing_cues = data.apply(get_opposing_cues, axis=1)
    
    # Compare the probability of choosing the TTB winner when there are few opposing cues (1-2)
    # versus when there are many opposing cues (5-6).
    p_few = chose_ttb[opposing_cues <= 2].mean()
    p_many = chose_ttb[opposing_cues >= 5].mean()
    
    if np.isnan(p_few) or np.isnan(p_many):
        return 0.0
        
    return float(p_few - p_many)
```

**Observed (real) value:** 0.0162 (var=0.0214)
**Predicted under pi_3:** -0.0275 (var=0.0130)
**Predicted under pi_5:** 0.0206 (var=0.0121)

## RESPONSE FORMAT

Return a JSON object with the following fields:
- interpretation: Freeform analysis of which theory better captured the observed data across both experiments, and why.
- verdict: Either 'new_model' (keep the same theory description, regenerate the predict / policy / parameter-ranges) or 'new_theory' (the current theory is degenerate; propose a brand-new theory).
- target_theory_idx: Which theory the verdict acts on: 1 or 2, matching THEORY 1 / THEORY 2. If verdict='new_model', this is the theory whose predict / policy / parameter-ranges should be regenerated (description preserved); if verdict='new_theory', this is the theory that will be replaced completely.
- recommendation: If verdict='new_model': how the current predict / policy / parameter-ranges should be revised. If verdict='new_theory': a sketch of the new theory that should be proposed and how it differs from both.


## Response

```json
{
  "interpretation": "Across almost all experiments (1 through 7), Theory 1 (Pure Take-The-Best with noise) consistently outperforms Theory 2 (Stochastic TTB). The empirical data shows a remarkable insensitivity to the number and validities of subordinate cues. For instance, in Experiment 1 and Experiment 2, the observed metric is much closer to the flat predictions of Theory 1 than the graded predictions of Theory 2. Theory 2's assumption that subjects sample among all discriminating features proportional to their validities leads to predictions that consistently miss the observed data, overestimating the effect of subordinate cues. Since Theory 1 is already a very strong non-compensatory baseline, Theory 2 should be replaced with a distinctly different theoretical framework to see if a compensatory or hybrid approach can capture the slight deviations from pure TTB.",
  "verdict": "new_theory",
  "target_theory_idx": 2,
  "recommendation": "Propose a completely new theory that provides a compensatory alternative to TTB. For example, a Weighted Additive (WADD) model with decision noise, or a dual-process mixture model where subjects use TTB on some trials and WADD on others. This will test whether the slight deviations from pure TTB observed in the data (e.g. in Experiment 8) are better explained by a compensatory mechanism rather than stochastic cue selection."
}
```

## Usage

```json
{
  "prompt_token_count": 10024,
  "candidates_token_count": 297,
  "total_token_count": 11188
}
```
