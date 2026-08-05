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
**Validities (n_features=7):** [0.9, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0, 0, 0]  B=[0, 1, 0, 0, 1, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 1, 0, 0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 1, 0, 1, 0]  B=[0, 1, 0, 0, 1, 0, 1]
  trial 5: A=[0, 1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 1]
  trial 7: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 0, 0, 1, 0]  B=[0, 0, 0, 1, 0, 0, 1]
  trial 9: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 0, 0, 1, 0, 1]  B=[1, 0, 0, 1, 0, 1, 0]

**Rationale:** This design quantitatively dissociates pure WADD from Evidence-Dependent Noise (EDN) by introducing 'symmetric conflict' trials. We use pairs of features with identical validities. By adding these pairs to opposing options, we increase the total feature conflict without changing the net weighted evidence difference (since the added validities perfectly cancel out, regardless of the gamma parameter). Pure WADD dictates that choice probability depends solely on the net weighted evidence, predicting identical choice probabilities across trials with the same base difference. In contrast, EDN posits that decision noise scales with total conflict, predicting that choice probabilities will regress toward 0.5 as symmetric conflict is added. This provides a parameter-free, strictly qualitative divergence between the theories.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Evidence-Dependent Noise: Decision-makers evaluate options using a single compensatory mechanism (Weighted Additive) where features are weighted by a subjective transformation of their validities. However, the decision process is subject to evidence-dependent noise: the temperature of the softmax choice rule scales with the total conflict between the options (defined as the total weighted evidence of features that differ between the two options). This ensures that trials with higher evidence magnitudes or greater feature conflict naturally generate higher decision noise. This single-mechanism approach preserves log-odds linearity while explaining why extremeness flattens across trials with varying evidence magnitudes.

**Parameters:**
- gamma: [0.0, 5.0]
- beta: [0.1, 20.0]
- theta: [0.0, 10.0]
- epsilon: [0.0, 0.1]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus expects shape (2, n_features); got {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Subjective feature weighting
    weights = val ** gamma
    
    # Calculate weighted sums of evidence for each option
    scores = np.dot(stim, weights)
    
    # Calculate conflict: total weight of features where the options differ
    diff = np.abs(stim[0] - stim[1])
    conflict = np.dot(diff, weights)
    
    # Effective beta scales inversely with conflict (higher conflict = more noise)
    beta_eff = beta / (1.0 + theta * conflict)
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta_eff * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Decision-makers integrate all available information by taking a weighted sum of each option's features, where the weights are subjective transformations of the cue validities. By exponentiating the raw validities by a free parameter gamma, the weighting scheme can smoothly interpolate between equal weighting (Tallying), proportional weighting (raw Weighted Additive), and lexicographic-like steep weighting (Take The Best). Choice probabilities are generated via a softmax over these subjectively weighted sums, combined with a lapse rate. Human behavior is best described by relatively flat (Tally-like) weights combined with substantial choice noise (lower beta).

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError(f"validities length {val.shape[0]} != n_features {stim.shape[1]}.")
        
    gamma = float(parameters["gamma"])
    weights = val ** gamma
    
    # Calculate the weighted sum of features for each option
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
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
[0] rationale: Evaluates the difference in the probability of choosing the option with higher net evidence when the total conflict is high versus when the total conflict is low. Under pure WADD, the net evidence is larger in the high conflict trials, so the probability should increase (positive difference). Under EDN, the decision noise scales with conflict, regressing choice probabilities toward 0.5, thus the difference should be negative or significantly lower.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Low conflict trials (net evidence 0.05)
    t1_mask = (data['A_tuple'] == (1,0,0,0,0,0)) & (data['B_tuple'] == (0,1,0,0,0,0))
    t6_mask = (data['A_tuple'] == (0,1,0,0,0,0)) & (data['B_tuple'] == (1,0,0,0,0,0))
    
    # High conflict trials (net evidence 0.15)
    t3_mask = (data['A_tuple'] == (1,0,1,0,1,0)) & (data['B_tuple'] == (0,1,0,1,0,1))
    t8_mask = (data['A_tuple'] == (0,1,0,1,0,1)) & (data['B_tuple'] == (1,0,1,0,1,0))
    
    p_t1 = (data[t1_mask]['response'] == 0).mean()
    p_t3 = (data[t3_mask]['response'] == 0).mean()
    
    p_t6 = (data[t6_mask]['response'] == 1).mean()
    p_t8 = (data[t8_mask]['response'] == 1).mean()
    
    # Average target probabilities
    p_high_conflict = np.nanmean([p_t3, p_t8])
    p_low_conflict = np.nanmean([p_t1, p_t6])
    
    if np.isnan(p_high_conflict) or np.isnan(p_low_conflict):
        return 0.0
        
    return float(p_high_conflict - p_low_conflict)
outcome: self_sim=-0.0017 (var=0.0137) adversary_sim=0.0758 (var=0.0150) welch_t=-2.284 p=0.02687 (N=25, alpha=0.01) -> reject

[1] rationale: By pooling the low and medium conflict trials to form a robust baseline and subtracting the accuracy on the highest conflict trials, we reduce per-subject variance while capturing the core divergence. Under pure WADD, accuracy monotonically increases with net evidence, so the high conflict (but highest net evidence) trials will be more accurate than the baseline, yielding a negative metric. Under EDN, the noise scales faster than the net evidence, causing a preference reversal where the highest conflict trials regress to chance, yielding a positive metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Low conflict trials (T1, T6)
    m_t1 = (data['A_tuple'] == (1,0,0,0,0,0)) & (data['B_tuple'] == (0,1,0,0,0,0))
    m_t6 = (data['A_tuple'] == (0,1,0,0,0,0)) & (data['B_tuple'] == (1,0,0,0,0,0))
    acc_t1 = (data[m_t1]['response'] == 0).mean()
    acc_t6 = (data[m_t6]['response'] == 1).mean()
    acc_low = np.nanmean([acc_t1, acc_t6])
    
    # Medium conflict trials (T2, T7)
    m_t2 = (data['A_tuple'] == (1,0,1,0,0,0)) & (data['B_tuple'] == (0,1,0,1,0,0))
    m_t7 = (data['A_tuple'] == (0,1,0,1,0,0)) & (data['B_tuple'] == (1,0,1,0,0,0))
    acc_t2 = (data[m_t2]['response'] == 0).mean()
    acc_t7 = (data[m_t7]['response'] == 1).mean()
    acc_med = np.nanmean([acc_t2, acc_t7])
    
    # High conflict trials (T3, T8)
    m_t3 = (data['A_tuple'] == (1,0,1,0,1,0)) & (data['B_tuple'] == (0,1,0,1,0,1))
    m_t8 = (data['A_tuple'] == (0,1,0,1,0,1)) & (data['B_tuple'] == (1,0,1,0,1,0))
    acc_t3 = (data[m_t3]['response'] == 0).mean()
    acc_t8 = (data[m_t8]['response'] == 1).mean()
    acc_high = np.nanmean([acc_t3, acc_t8])
    
    if np.isnan(acc_low) or np.isnan(acc_med) or np.isnan(acc_high):
        return 0.0
        
    # Compare pooled lower-conflict accuracy vs highest-conflict accuracy
    return float((acc_low + acc_med) / 2.0 - acc_high)
outcome: self_sim=-0.0004 (var=0.0125) adversary_sim=-0.0617 (var=0.0119) welch_t=+1.957 p=0.05617 (N=25, alpha=0.01) -> reject

[2] rationale: Compares accuracy on a trial with lower conflict but higher net evidence (T4) against trials with higher conflict but slightly lower net evidence (T3/T8). Under pure WADD, the accuracy difference is driven solely by the small +0.05 difference in net evidence, producing a small positive gap. Under EDN, T3/T8 are heavily penalized by their extreme conflict (4.65 vs 3.30), suppressing their accuracy significantly, which results in a much larger positive gap for this metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # High conflict, moderate net evidence (T3 & T8)
    # Net evidence = +0.15, Conflict = 4.65
    m_t3 = (data['A_tuple'] == (1,0,1,0,1,0)) & (data['B_tuple'] == (0,1,0,1,0,1))
    m_t8 = (data['A_tuple'] == (0,1,0,1,0,1)) & (data['B_tuple'] == (1,0,1,0,1,0))
    acc_t3 = (data[m_t3]['response'] == 0).mean()
    acc_t8 = (data[m_t8]['response'] == 1).mean()
    acc_high_conflict = np.nanmean([acc_t3, acc_t8])
    
    # Lower conflict, higher net evidence (T4)
    # Net evidence = +0.20, Conflict = 3.30
    m_t4 = (data['A_tuple'] == (1,1,0,0,0,0)) & (data['B_tuple'] == (0,0,1,1,0,0))
    acc_lower_conflict = (data[m_t4]['response'] == 0).mean()
    
    if np.isnan(acc_high_conflict) or np.isnan(acc_lower_conflict):
        return 0.0
        
    return float(acc_lower_conflict - acc_high_conflict)
outcome: self_sim=0.0625 (var=0.0196) adversary_sim=0.0142 (var=0.0248) welch_t=+1.147 p=0.257 (N=25, alpha=0.01) -> reject

[3] rationale: Measures the accuracy gain from the lowest evidence/lowest conflict trials (T1, T6) to the highest evidence/highest conflict trial (T5). Under pure WADD, T5 has 9x the net evidence of T1/T6, leading to a massive increase in accuracy. Under Evidence-Dependent Noise (EDN), T5 also has nearly 3x the total conflict, which drastically increases decision noise and pulls T5's accuracy back down toward chance. This creates a much smaller accuracy gap under EDN compared to WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Trial 5: Highest net evidence (+0.45) but also highest conflict (4.65)
    m_t5 = (data['A_tuple'] == (1, 1, 1, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 1, 1))
    acc_t5 = (data[m_t5]['response'] == 0).mean()
    
    # Trials 1 & 6: Lowest net evidence (+0.05) and lowest conflict (1.75)
    m_t1 = (data['A_tuple'] == (1, 0, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 0, 0, 0, 0))
    m_t6 = (data['A_tuple'] == (0, 1, 0, 0, 0, 0)) & (data['B_tuple'] == (1, 0, 0, 0, 0, 0))
    
    acc_t1 = (data[m_t1]['response'] == 0).mean()
    acc_t6 = (data[m_t6]['response'] == 1).mean()
    acc_low_conflict = np.nanmean([acc_t1, acc_t6])
    
    if np.isnan(acc_t5) or np.isnan(acc_low_conflict):
        return 0.0
        
    return float(acc_t5 - acc_low_conflict)
outcome: self_sim=0.0758 (var=0.0228) adversary_sim=0.1650 (var=0.0327) welch_t=-1.892 p=0.06473 (N=25, alpha=0.01) -> reject

[4] rationale: This metric leverages a parameter-free qualitative dissociation. It pairs trials that have EXACTLY the same net weighted evidence difference under any possible gamma parameter (because the added features on both sides have identical validities). Under pure WADD, choice probabilities depend solely on the net evidence, so the expected difference in accuracy between the lower-conflict and higher-conflict trial in each pair is precisely zero. Under Evidence-Dependent Noise (EDN), the added symmetric features increase total conflict without changing net evidence, which scales up decision noise and drives accuracy toward 0.5. Thus, EDN uniquely predicts a strictly positive drop in accuracy across these matched pairs.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # Pair 1: T1 vs T4 (Net evidence: v0 - v1)
    m_t1 = (data['A_tuple'] == (1, 0, 0, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 1, 0, 0, 0, 0, 0))
    m_t4 = (data['A_tuple'] == (1, 0, 0, 1, 0, 1, 0)) & (data['B_tuple'] == (0, 1, 0, 0, 1, 0, 1))
    p_t1 = (data[m_t1]['response'] == 0).mean()
    p_t4 = (data[m_t4]['response'] == 0).mean()
    diff1 = p_t1 - p_t4
    
    # Pair 2: T9 vs T10 (Net evidence: v1 - v0, B is better)
    m_t9 = (data['A_tuple'] == (0, 1, 0, 0, 0, 0, 0)) & (data['B_tuple'] == (1, 0, 0, 0, 0, 0, 0))
    m_t10 = (data['A_tuple'] == (0, 1, 0, 0, 1, 0, 1)) & (data['B_tuple'] == (1, 0, 0, 1, 0, 1, 0))
    p_t9 = (data[m_t9]['response'] == 1).mean()
    p_t10 = (data[m_t10]['response'] == 1).mean()
    diff2 = p_t9 - p_t10
    
    # Pair 3: T5 vs T6 (Net evidence: v1 - v3)
    m_t5 = (data['A_tuple'] == (0, 1, 0, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 0, 0, 0))
    m_t6 = (data['A_tuple'] == (0, 1, 0, 0, 0, 1, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 0, 0, 1))
    p_t5 = (data[m_t5]['response'] == 0).mean()
    p_t6 = (data[m_t6]['response'] == 0).mean()
    diff3 = p_t5 - p_t6
    
    # Pair 4: T7 vs T8 (Net evidence: v0 - v3)
    m_t7 = (data['A_tuple'] == (1, 0, 0, 0, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 0, 0, 0))
    m_t8 = (data['A_tuple'] == (1, 0, 0, 0, 0, 1, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 0, 0, 1))
    p_t7 = (data[m_t7]['response'] == 0).mean()
    p_t8 = (data[m_t8]['response'] == 0).mean()
    diff4 = p_t7 - p_t8
    
    diffs = [d for d in [diff1, diff2, diff3, diff4] if not np.isnan(d)]
    
    if len(diffs) == 0:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=-0.0094 (var=0.0143) adversary_sim=0.0117 (var=0.0082) welch_t=-0.704 p=0.4852 (N=25, alpha=0.01) -> reject

[5] rationale: Given the parameter distributions of the two theories, they make vastly different predictions about overall choice accuracy. The Competing Theory (WADD) models high baseline noise (epsilon up to 0.5) and relatively low sensitivity (beta up to 5.0), predicting that subjects will frequently lapse or choose randomly, leading to an overall accuracy heavily suppressed toward 0.5. In contrast, the Advocated Theory (EDN) posits a much lower base lapse rate (epsilon up to 0.1) and higher sensitivity (beta up to 20.0), with noise instead being driven dynamically by feature conflict. Averaged across the 90 trials, EDN predicts a significantly higher and more stable overall accuracy than WADD. This simple, robust metric avoids the high variance associated with trial-specific subtractions while capturing a massive divergence in the global predictability of choices.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    data = data.copy()
    data['A_tuple'] = data['option_a_ratings'].apply(tuple)
    data['B_tuple'] = data['option_b_ratings'].apply(tuple)
    
    # In this design, Option A has higher net evidence in all trials EXCEPT T9 and T10.
    is_t9 = (data['A_tuple'] == (0, 1, 0, 0, 0, 0, 0)) & (data['B_tuple'] == (1, 0, 0, 0, 0, 0, 0))
    is_t10 = (data['A_tuple'] == (0, 1, 0, 0, 1, 0, 1)) & (data['B_tuple'] == (1, 0, 0, 1, 0, 1, 0))
    
    b_better = is_t9 | is_t10
    
    # Correct choice is B (1) for T9/T10, and A (0) for all other trials
    correct_response = np.where(b_better, 1, 0)
    
    # Calculate overall accuracy
    accuracy = (data['response'] == correct_response).mean()
    
    return float(accuracy)
outcome: self_sim=0.5856 (var=0.0115) adversary_sim=0.5382 (var=0.0046) welch_t=+1.864 p=0.06957 (N=25, alpha=0.01) -> reject

[6] rationale: This metric measures the difference in accuracy between trials where the net evidence comes from the top two validities (0.9 vs 0.8) and trials where it comes from lower validities (0.8 vs 0.7). Under the Competing Theory (WADD), gamma is relatively low (mean ~1), meaning the net evidence in both cases is approximately 0.1, leading to similar accuracy and a difference near zero. Under the Advocated Theory (EDN), gamma is much higher (up to 5.0). Due to the convex exponentiation, the subjective difference between 0.9 and 0.8 is significantly larger than the difference between 0.8 and 0.7. Consequently, EDN predicts substantially higher accuracy for the 0.9 vs 0.8 trials compared to the 0.8 vs 0.7 trial, yielding a strongly positive gap.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    data = data.copy()
    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    # T1: A=1000000, B=0100000 (0.9 vs 0.8) -> correct A
    m_t1 = (data['A_str'] == '1000000') & (data['B_str'] == '0100000')
    acc_t1 = (data[m_t1]['response'] == 0).mean()
    
    # T9: A=0100000, B=1000000 (0.8 vs 0.9) -> correct B
    m_t9 = (data['A_str'] == '0100000') & (data['B_str'] == '1000000')
    acc_t9 = (data[m_t9]['response'] == 1).mean()
    
    # T5: A=0100000, B=0001000 (0.8 vs 0.7) -> correct A
    m_t5 = (data['A_str'] == '0100000') & (data['B_str'] == '0001000')
    acc_t5 = (data[m_t5]['response'] == 0).mean()
    
    acc_high_val_diff = np.nanmean([acc_t1, acc_t9])
    
    if np.isnan(acc_high_val_diff) or np.isnan(acc_t5):
        return 0.0
        
    return float(acc_high_val_diff - acc_t5)
outcome: self_sim=-0.0233 (var=0.0385) adversary_sim=0.0178 (var=0.0398) welch_t=-0.734 p=0.4663 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    data = data.copy()\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    # High evidence, low conflict trials (T7, T8)\n    m_t7 = (data['A_str'] == '1000000') & (data['B_str'] == '0001000')\n    m_t8 = (data['A_str'] == '1000010') & (data['B_str'] == '0001001')\n    acc_t7 = (data[m_t7]['response'] == 0).mean()\n    acc_t8 = (data[m_t8]['response'] == 0).mean()\n    acc_high_ev = np.nanmean([acc_t7, acc_t8])\n    \n    # Low evidence, high conflict trials (T2, T3, T4, T10)\n    m_t2 = (data['A_str'] == '1001000') & (data['B_str'] == '0100100')\n    m_t3 = (data['A_str'] == '1000010') & (data['B_str'] == '0100001')\n    m_t4 = (data['A_str'] == '1001010') & (data['B_str'] == '0100101')\n    m_t10 = (data['A_str'] == '0100101') & (data['B_str'] == '1001010')\n    \n    acc_t2 = (data[m_t2]['response'] == 0).mean()\n    acc_t3 = (data[m_t3]['response'] == 0).mean()\n    acc_t4 = (data[m_t4]['response'] == 0).mean()\n    acc_t10 = (data[m_t10]['response'] == 1).mean()\n    acc_high_conf = np.nanmean([acc_t2, acc_t3, acc_t4, acc_t10])\n    \n    if np.isnan(acc_high_ev) or np.isnan(acc_high_conf):\n        return 0.0\n        \n    return float(acc_high_ev - acc_high_conf)",
  "rationale": "This metric contrasts trials with the highest net evidence and lowest conflict (T7, T8) against trials with the lowest net evidence and highest conflict (T2, T3, T4, T10). Under pure WADD, the accuracy difference is driven solely by the moderate gap in net evidence, producing a small positive difference. Under EDN, the combination of a convex weighting function (amplifying the net evidence in T7/T8) and massive conflict penalties (suppressing accuracy in T2/T3/T4/T10) produces a dramatically larger positive gap. By pooling multiple trials for each condition, we significantly reduce per-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 8928,
  "candidates_token_count": 808,
  "total_token_count": 14034
}
```
