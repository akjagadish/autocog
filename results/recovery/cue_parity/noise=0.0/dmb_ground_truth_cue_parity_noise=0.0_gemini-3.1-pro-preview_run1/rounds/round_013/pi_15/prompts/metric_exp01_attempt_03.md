# metric_exp01_attempt_03

## System Prompt

You are a psychology researcher proposing a metric in the Decision Making (Binary Features) domain.

Your goal is adversarial: propose a metric that DISCRIMINATES the two theories — i.e., its value, computed on data simulated under your advocated theory, should be as far as possible from its value computed on data simulated under the competing theory. The direction of the gap does not matter; what matters is that the two theories produce visibly different numbers on this metric. The metric is computed on the data collected from the experimental design provided in the prompt. Produce a metric where you're prediction will be much more accurate than the competing theory's prediction on human data.

Your metric is a Python function

    metric(data: pd.DataFrame) -> float

Available imports inside `metric`:
- numpy as np
- pandas as pd

The system evaluates your metric in two ways and reports the pair as `point_estimate (var=between_subject_variance)` everywhere downstream:
- `point_estimate` is `metric(data)` applied to the FULL pooled DataFrame (all subjects together) — the canonical scalar;
- `between_subject_variance` is the population variance (`ddof=0`) of `metric(subj_df)` re-applied per `subject_id`, summarising how stable the metric is across subjects. If your metric only makes sense on multi-subject data this will fall back to `n/a` and the metric is rejected (the acceptance test below cannot run without it). Prefer metrics that work both on the pooled DataFrame and on a single subject's slice.

Acceptance rule: the system simulates each theory and runs Welch's two-sample t-test on `(point_estimate_self, between_subject_variance_self, N)` vs. `(point_estimate_adv, between_subject_variance_adv, N)`, where N is the number of HUMAN subjects the experiment will actually be run with (a fixed small number, currently 25). Your metric is admitted iff the two-sided p-value is below the significance level (currently alpha=0.01). Implication: a large between-theory gap is NOT enough — if either theory's metric is also highly variable across subjects, N humans won't reliably distinguish them and the metric will be rejected. Aim for contrasts that are both large in mean AND tight per subject.

Do NOT propose metrics that are trivially true for your theory.


## User Prompt

## EXPERIMENTAL DOMAIN
Subjects repeatedly choose between two fictitious products, A and B. Each option is described by a vector of binary expert ratings (each 0 or 1). Every experiment fixes its own feature count (via `validities` length) and per-expert validities; both are LLM-proposed. The validities are communicated to the subject in the instructions. Subjects pick whichever product they believe is of higher quality. There is no trial-by-trial correctness feedback.

Each subject completes ~96 trials in a single block, with order randomized independently per subject. On every trial the subject sees two options A and B, each described by `n_features` binary expert ratings (each 0 or 1). The per-feature validities and n_features are fixed per experiment (design-time choices). Validities are communicated to the subject in the instructions. Both `n_features` and `validities` are exposed to your `predict` via the `parameters` dict. The subject chooses A or B; no correctness feedback is provided after the choice.

## CHOSEN EXPERIMENTAL DESIGN
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 1]
  trial 2: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the Advocated Theory (Weighted Sequential Evidence Accumulation with Self-Excitation/Decay) from the Competing Theory (Unweighted Sequential Evidence Accumulation with a stopping rule). The Competing Theory maintains a running sum of unweighted differences (+1 or -1) and stops if a threshold is hit. By constructing trials where the unweighted running tally never reaches a positive threshold and ends negative (e.g., -1, 0, -1, 0, -1), the Competing Theory is structurally forced to consistently predict one option (B). Meanwhile, the Advocated Theory weights evidence by validities and allows previous evidence to either decay (leak > 0) or self-excite (leak < 0). With a high leak (rapid forgetting), the Advocated Theory can stop early on a subsequent feature and predict the exact opposite option (A). Furthermore, self-excitation allows the Advocated Theory to massively amplify initial favorable evidence such that a multitude of later opposing features cannot overcome it, creating distinct divergent predictions across parameter regimes.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Sequential Evidence Accumulation with Self-Excitation/Decay (Leaky Accumulator). Decision-makers process information sequentially in descending order of feature validity. Each feature's evidence is weighted by its validity and added to a running accumulator. Crucially, previously accumulated evidence can either 'leak' (decay) or self-excite (amplify) as new features are evaluated. A choice is made either when the accumulator hits a predefined threshold or when all features are exhausted. Self-excitation allows the model to capture strong primacy effects even when the decision threshold is not reached.

**Parameters:**
- threshold: [0.0, 5.0]
- leak: [-1.0, 1.0]
- gamma: [0.0, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    leak = float(parameters["leak"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Evaluate features in descending order of validity
    order = np.argsort(val)[::-1]
    
    A = 0.0
    for idx in order:
        diff = a[idx] - b[idx]
        weight = val[idx] ** gamma
        
        # Leaky accumulation (or self-excitation if leak < 0)
        A = A * (1.0 - leak) + weight * diff
        
        # Stopping rule
        if abs(A) >= threshold and abs(A) > 1e-9:
            break
            
    scores = np.array([A, -A])
    
    # Softmax conversion to probabilities
    z = beta * scores
    z -= np.max(z)  # For numerical stability
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Decision-makers use a Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity, maintaining a running sum of the differences between the options. If the absolute accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally. This allows the model to smoothly transition between Take-The-Best (low threshold) and Tallying (high threshold) behaviors.

**Parameters:**
- threshold: [0.0, 3.0]
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## DATA SCHEMA
Your metric receives a tidy per-trial pandas DataFrame stacking all subjects (rows grouped by `subject_id`, in trial order). Columns:
- subject_id: Subject identifier (one row per trial per subject).
- option_a_ratings: List of n_features binary expert ratings (each 0 or 1) for option A on this trial.
- option_b_ratings: List of n_features binary expert ratings (each 0 or 1) for option B on this trial.
- response: 0 if subject chose A, 1 if subject chose B.

## IMPLEMENTATION GUARDRAILS
Any column in the schema above whose description names a list / tuple / np.ndarray (i.e. a per-trial sequence of values) holds non-scalar cells. Those cells are NOT hashable, so operations that hash row values fail with `TypeError: unhashable type: 'list'`. Treating `<seq_col>` as a placeholder for any such sequence-valued column:
- Avoid: `data.groupby('<seq_col>')`, `data['<seq_col>'].value_counts()`,     `data['<seq_col>'].nunique()`, `data['<seq_col>'].unique()` (returns     an object array but downstream `set()` / `in dict` will crash),     `set(data['<seq_col>'])`, `data['<seq_col>'].isin([...])` against list     values, or using a list cell as a dict key.
- If you need a hashable surrogate, project to one first, e.g.:
    - `data['<seq_col>_key'] = data['<seq_col>'].apply(tuple)` then group by `<seq_col>_key`
    - `data['<seq_col>_str'] = data['<seq_col>'].apply(lambda x: ''.join(map(str, x)))`
    Scalar columns (ints, floats, strings like `subject_id`, integer     responses, etc.) hash fine and can be used directly.
- Generator expressions inside function calls like `map()` or `join()` MUST be     parenthesized. For example:
    - WRONG: `map(str, int(v) for v in x)` → SyntaxError
    - RIGHT: `map(str, (int(v) for v in x))` or use a list comp: `[str(int(v)) for v in x]`
- Always verify your code is syntactically valid Python before returning it.

## METRICS YOU ALREADY TRIED AND FAILED ON
Each entry below is a metric you previously proposed in this round that did NOT discriminate the two theories at the human sample size — either it errored, its between-subject variance was unavailable, or Welch's t-test on `(self mean, self var, N)` vs. `(adv mean, adv var, N)` returned p ≥ alpha. The `outcome` line is the simulation result (means, between-subject variances, t-statistic and p-value at the human N) on the same `data_self` / `data_adv` your next metric will be evaluated on. Use the numbers to see where your hypothesised contrast collapsed — small mean gap, large per-subject variance, or both — and propose something qualitatively different. Don't repeat the same idea with cosmetic tweaks.
[0] rationale: On Trial 4, the unweighted difference between features favors B at the first feature (-1), then moves towards A but never reaches a positive sum (+1, +1), and finally ends strictly negative (-1, -1). As a result, the Competing Theory (which uses unweighted evidence accumulation) will ALWAYS predict Option B, regardless of its threshold parameter (it either stops immediately at feature 1, or tallies to the end and finds a negative sum). Its only chance of picking A is through the random epsilon error. In contrast, the Advocated Theory with a positive 'leak' parameter will quickly forget the initial negative evidence from feature 1. The subsequent favorable evidence from features 2 and 3 will dominate the accumulator, allowing it to hit a positive threshold and confidently choose Option A. This metric isolates that specific divergence by measuring the proportion of A choices on Trial 4.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Identify Trial 4 where A = [0, 1, 1, 0, 0] and B = [1, 0, 0, 1, 1]
    is_trial_4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    if is_trial_4.sum() == 0:
        return 0.0
        
    # Calculate the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[is_trial_4, 'response'] == 0))
outcome: self_sim=0.1750 (var=0.0369) adversary_sim=0.1525 (var=0.0135) welch_t=+0.501 p=0.6189 (N=25, alpha=0.01) -> reject

[1] rationale: On Trial 6, the unweighted difference sequence is [+1, -1, +1, +1, +1]. The Competing Theory (unweighted tallying) will always accumulate positive or zero evidence (+1, 0, +1, +2, +3). Because the running tally never goes negative, it can never hit a negative threshold, meaning the Competing Theory will almost always choose Option A (only deviating due to random epsilon errors). In contrast, the Advocated Theory can choose Option B: if the 'leak' parameter is high, the initial +1 evidence from the first feature decays rapidly, and the -1 evidence from the second feature (which has high validity, 0.9) can push the accumulator past a negative threshold, stopping the process early and resulting in a choice for Option B. Thus, measuring the choice proportion for Option A on Trial 6 will yield a value near 1.0 for the Competing Theory, but significantly lower for the Advocated Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Identify Trial 6 where A = [1, 0, 1, 1, 1] and B = [0, 1, 0, 0, 0]
    # The difference is [+1, -1, +1, +1, +1]
    is_trial_6 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 1, 1))
    if is_trial_6.sum() == 0:
        return 0.0
        
    # Calculate the proportion of times Option A was chosen (response == 0)
    return float(np.mean(data.loc[is_trial_6, 'response'] == 0))
outcome: self_sim=0.8675 (var=0.0099) adversary_sim=0.8738 (var=0.0076) welch_t=-0.236 p=0.8144 (N=25, alpha=0.01) -> reject

[2] rationale: This metric measures the sum of the probabilities of choosing Option A on Trial 2 and Trial 5. For the Advocated theory, these two trials perfectly anti-correlate with the 'leak' parameter. If leak < 0 (self-excitation), the first feature dominates: T2's first feature favors A, so it chooses A; T5's first feature favors B, so it chooses B. If leak > 0 (decay), the first feature is forgotten and later features dominate: T2's later features favor B, so it chooses B; T5's later features favor A, so it chooses A. In either case, the subject chooses A on exactly one of the two trials, making the sum P(A|T2) + P(A|T5) converge tightly to 1.0 regardless of the subject's specific parameters. Even random epsilon errors sum to 1.0 (0.5 + 0.5), preserving this invariant perfectly and ensuring near-zero between-subject variance. For the Competing theory, the sum depends on the unweighted stopping threshold. T2 chooses A if threshold <= 1, and T5 chooses A if threshold > 2. These conditions are mutually exclusive. Thus, the sum is 1 if threshold <= 1 or threshold > 2, but it is exactly 0 if the threshold is between 1 and 2. Since the threshold is uniformly sampled from [0, 3], about 33% of subjects will have a sum of 0, dropping the population mean significantly below 1.0. This creates a highly significant mean difference between the two theories, while the Advocated theory maintains extreme stability.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify Trial 2 where A = [1, 0, 0, 0, 0]
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    
    # Identify Trial 5 where A = [0, 0, 1, 1, 1]
    is_t5 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))
    
    # Calculate the proportion of times Option A was chosen (response == 0) for both trials
    p_a_t2 = float(np.mean(data.loc[is_t2, 'response'] == 0)) if is_t2.sum() > 0 else 0.0
    p_a_t5 = float(np.mean(data.loc[is_t5, 'response'] == 0)) if is_t5.sum() > 0 else 0.0
    
    # Return the sum of probabilities
    return p_a_t2 + p_a_t5
outcome: self_sim=0.7250 (var=0.1345) adversary_sim=0.8100 (var=0.1139) welch_t=-0.853 p=0.3981 (N=25, alpha=0.01) -> reject

[3] rationale: This metric combines the choice probabilities on Trial 2 and Trial 4 to maximize the divergence between the two theories. In the Competing theory (unweighted tallying), Option A is NEVER chosen on Trial 4 because the first feature drops the unweighted sum to -1, meaning it will either stop immediately (if threshold <= 1) or tally to a final negative sum. Thus, P(A|T4) is strictly ~0.0. In contrast, the Advocated theory (leaky accumulation) CAN choose Option A on Trial 4: if the decay is weak, the initial negative evidence from feature 1 decays, allowing the positive evidence from features 2 and 3 to push the accumulator positive and hit the threshold. By weighting P(A|T4) heavily and adding it to P(A|T2) (which is mutually exclusive with T4 in the Advocated theory), we selectively boost the Advocated theory's mean score while the Competing theory's score remains low and tightly bounded around 0.33.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Identify Trial 2 where A = [1, 0, 0, 0, 0]
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    
    # Identify Trial 4 where A = [0, 1, 1, 0, 0]
    is_t4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_t2 = float(np.mean(data.loc[is_t2, 'response'] == 0)) if is_t2.sum() > 0 else 0.0
    p_a_t4 = float(np.mean(data.loc[is_t4, 'response'] == 0)) if is_t4.sum() > 0 else 0.0
    
    # Weight Trial 4 heavily to amplify the difference between the theories
    return p_a_t2 + 3.0 * p_a_t4
outcome: self_sim=0.9212 (var=0.3656) adversary_sim=0.8237 (var=0.2756) welch_t=+0.609 p=0.5456 (N=25, alpha=0.01) -> reject

[4] rationale: For the Competing theory (unweighted tallying), Trial 2 never reaches a negative running sum (diff: +1, 0, +1, 0, +1), so it ALWAYS chooses Option A, regardless of threshold. Trial 5 (diff: +1, +2, +1, 0, -1) chooses A if the threshold is <= 2, but defaults to B if the threshold > 2 (since the final tally is -1). Thus, P(A|T2) - P(A|T5) for the Competing theory is strictly non-negative, averaging ~0.33 because roughly a third of subjects have a threshold > 2. 

For the Advocated theory (leaky accumulation), both trials start with favorable evidence for A. If self-excitation dominates (leak < 0), both trials confidently choose A, yielding a difference of ~0. If decay dominates (leak > 0), Trial 5 provides strong reinforcing evidence for A (second feature is also +1), while Trial 2 provides contradictory evidence (second feature is -1), causing Trial 2 to choose A less frequently than Trial 5. Thus, P(A|T2) - P(A|T5) for the Advocated theory is strictly non-positive, averaging < 0. 

This structural divergence creates a robust mean gap (~0.33 vs <0) with low between-subject variance, strongly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify Trial 2 where A = [1, 0, 1, 0, 1]
    is_t2 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 1, 0, 1))
    
    # Identify Trial 5 where A = [1, 1, 0, 0, 0]
    is_t5 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0))
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_t2 = float(np.mean(data.loc[is_t2, 'response'] == 0)) if is_t2.sum() > 0 else 0.0
    p_a_t5 = float(np.mean(data.loc[is_t5, 'response'] == 0)) if is_t5.sum() > 0 else 0.0
    
    return p_a_t2 - p_a_t5
outcome: self_sim=0.1083 (var=0.1165) adversary_sim=0.2283 (var=0.1391) welch_t=-1.187 p=0.2412 (N=25, alpha=0.01) -> reject

[5] rationale: This metric leverages a structural invariant of the Competing Theory (unweighted tallying with a stopping rule). For the Competing Theory, Trial 3 and Trial 7 produce the exact same unweighted running sum sequence at all critical thresholds (both choose A if threshold <= 1, and both tally to a negative sum and choose B if threshold > 1). Thus, P(A|T7) - P(A|T3) is strictly 0. Similarly, Trial 4 and Trial 8 produce identical outcomes for the Competing Theory (both choose B if threshold <= 1, and both tally to a positive sum and choose A if threshold > 1), meaning P(A|T4) - P(A|T8) is strictly 0. Consequently, the combined metric (P(A|T4) - P(A|T8)) + (P(A|T7) - P(A|T3)) is exactly 0 for the Competing Theory, with near-zero between-subject variance (only epsilon noise). In contrast, for the Advocated Theory (leaky accumulation), Trial 4 and Trial 7 have favorable evidence at feature 4 (+1), whereas Trial 8 and Trial 3 have negative evidence at feature 4 (-1). For subjects with a positive leak (decay), this difference at feature 4 significantly influences the final decision, making P(A|T4) > P(A|T8) and P(A|T7) > P(A|T3). This produces a strictly positive mean for the Advocated Theory, perfectly discriminating the two models.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify Trial 3 where A = [1, 0, 0, 0, 0]
    is_t3 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 0, 0))
    # Identify Trial 4 where A = [0, 1, 1, 1, 1]
    is_t4 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 1, 1))
    # Identify Trial 7 where A = [1, 0, 0, 1, 0]
    is_t7 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 0))
    # Identify Trial 8 where A = [0, 1, 1, 0, 0]
    is_t8 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_t3 = float(np.mean(data.loc[is_t3, 'response'] == 0)) if is_t3.sum() > 0 else 0.0
    p_a_t4 = float(np.mean(data.loc[is_t4, 'response'] == 0)) if is_t4.sum() > 0 else 0.0
    p_a_t7 = float(np.mean(data.loc[is_t7, 'response'] == 0)) if is_t7.sum() > 0 else 0.0
    p_a_t8 = float(np.mean(data.loc[is_t8, 'response'] == 0)) if is_t8.sum() > 0 else 0.0
    
    # The metric isolates the effect of feature 4 across two pairs of trials
    return (p_a_t4 - p_a_t8) + (p_a_t7 - p_a_t3)
outcome: self_sim=0.5517 (var=0.2686) adversary_sim=0.4650 (var=0.1784) welch_t=+0.648 p=0.5201 (N=25, alpha=0.01) -> reject

[6] rationale: This metric evaluates the sum of the probabilities of choosing Option A on Trial 7 and Trial 8. For the Competing Theory (unweighted tallying), Trial 7's running sum of differences is [+1, 0, -1, 0, -1]. It only chooses A if the stopping threshold is <= 1 (stopping at the first feature). Trial 8's running sum is [-1, 0, +1, 0, -1]. It stops at the first feature and chooses B if the threshold is <= 1, and otherwise finishes with a negative tally and chooses B. Thus, P(A|T8) is strictly 0. The sum P(A|T7) + P(A|T8) is therefore exactly 1 if threshold <= 1, and exactly 0 if threshold > 1. Given the uniform threshold distribution [0, 3], the Competing Theory averages ~0.33 with high between-subject variance. 

For the Advocated Theory (leaky accumulation), Trial 7 and Trial 8 are near-perfect opposites. If the leak parameter is negative (self-excitation/primacy), the first feature dominates: T7 chooses A and T8 chooses B. If the leak parameter is positive (decay/recency), the later features dominate: T7's negative later features lead to B, while T8's positive features (features 2 and 3) lead to A. In all parameter regimes, the Advocated Theory strongly prefers A on exactly one of the two trials, causing the sum P(A|T7) + P(A|T8) to be invariant at ~1.0. Even random epsilon errors perfectly cancel out in this sum (e.g., 0.75 + 0.25 = 1.0). This structural invariant ensures the Advocated Theory yields a mean of ~1.0 with near-zero between-subject variance, creating a massive and highly significant gap against the Competing Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify Trial 7 where A = [1, 0, 0, 1, 0]
    is_t7 = data['option_a_ratings'].apply(lambda x: tuple(x) == (1, 0, 0, 1, 0))
    
    # Identify Trial 8 where A = [0, 1, 1, 0, 0]
    is_t8 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))
    
    # Calculate the proportion of times Option A was chosen (response == 0)
    p_a_t7 = float(np.mean(data.loc[is_t7, 'response'] == 0)) if is_t7.sum() > 0 else 0.0
    p_a_t8 = float(np.mean(data.loc[is_t8, 'response'] == 0)) if is_t8.sum() > 0 else 0.0
    
    return p_a_t7 + p_a_t8
outcome: self_sim=0.6783 (var=0.1436) adversary_sim=0.5067 (var=0.1625) welch_t=+1.551 p=0.1274 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Identify Trial 1 where A = [0, 1, 0, 1, 0] and B = [1, 0, 1, 0, 1]\n    is_t1 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 0))\n    \n    # Identify Trial 8 where A = [0, 1, 1, 0, 0] and B = [1, 0, 0, 1, 1]\n    is_t8 = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 1, 0, 0))\n    \n    # Calculate the proportion of times Option A was chosen (response == 0)\n    p_a_t1 = float(np.mean(data.loc[is_t1, 'response'] == 0)) if is_t1.sum() > 0 else 0.0\n    p_a_t8 = float(np.mean(data.loc[is_t8, 'response'] == 0)) if is_t8.sum() > 0 else 0.0\n    \n    return p_a_t8 - p_a_t1",
  "rationale": "For the Competing Theory (unweighted tallying with a stopping rule), Trial 1 and Trial 8 produce the exact same accumulated evidence sequence relative to any stopping threshold. In both trials, the initial feature drops the tally to -1. If the threshold is <= 1, the process stops immediately at -1. If the threshold is > 1, both trials eventually tally to a final sum of exactly -1. Consequently, the Competing Theory produces identical choice probabilities for both trials, meaning P(A|T8) - P(A|T1) is exactly 0 with near-zero between-subject variance (only binomial sampling noise). In contrast, the Advocated Theory (leaky accumulation) differentiates the two. Trial 8 features positive evidence at the highly valid 3rd feature, whereas Trial 1 has negative evidence there. For subjects with a decay parameter (leak > 0), the initial negative evidence decays, allowing Trial 8's positive 3rd feature to occasionally push the accumulator past a positive threshold, resulting in an Option A choice. Trial 1, lacking this strong mid-sequence positive evidence, remains strictly negative and chooses B. This structural divergence guarantees a strictly positive mean for the Advocated Theory, creating a highly significant statistical gap against the invariant 0.0 of the Competing Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 8142,
  "candidates_token_count": 617,
  "total_token_count": 39184
}
```
