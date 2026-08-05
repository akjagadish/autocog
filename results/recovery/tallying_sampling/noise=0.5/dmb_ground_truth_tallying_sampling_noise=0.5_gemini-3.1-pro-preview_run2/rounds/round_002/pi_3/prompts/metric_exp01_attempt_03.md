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
**Validities (n_features=5):** [0.95, 0.9, 0.55, 0.5, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 2: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate pure Tallying (Equal Weight) from Tallying with Validity Bias, this design uses a highly skewed validity distribution. This allows us to construct two types of critical trials: 'Tally Ties' where both options have the same number of positive features but differ significantly in total validity, and 'Conflict Trials' where one option has more positive features (higher tally) but the other option has a higher total validity. Pure Tallying predicts exact indifference on all Tally Ties and bases choices on Conflict Trials strictly on the feature count. In contrast, the Competing Theory predicts that the validity bias will break Tally Ties in favor of the higher-validity option, and will attenuate (or even reverse) the choice probability on Conflict Trials.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make decisions between options using a Tallying (Equal Weight) heuristic. Instead of weighting features by their validities or relying on a single discriminating cue, decision-makers simply count the number of positive features for each option. They choose the option with the higher total count, treating all cues as equally important. When counts are tied, they guess. Response noise is modeled via a softmax over the tally scores and a uniform lapse rate. The choice is relatively noisy, preventing the strategy from becoming perfectly deterministic even when one option has a clear tally advantage.

**Parameters:**
- beta: [0.1, 1.5]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: compute the sum of features for each option (equal weighting)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Tallying with Validity Bias: People primarily make decisions by tallying the number of positive features for each option (treating all cues as roughly equal), but they also incorporate a secondary, subtle bias based on the explicit cue validities. This means that while the overall count of positive features dominates the choice, the validities of those features can break ties or slightly adjust the strength of preference, blending a compensatory equal-weight strategy with a sensitivity to information quality.

**Parameters:**
- beta: [0.1, 3.0]
- epsilon: [0.0, 0.5]
- w_val: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    # Tallying component: equal weight sum
    tally_scores = stim.sum(axis=1)
    
    # Validity component: weighted sum using explicit validities
    validities = np.asarray(parameters["validities"], dtype=float)
    val_scores = stim @ validities
    
    # Blend the two strategies
    w_val = float(parameters["w_val"])
    scores = (1.0 - w_val) * tally_scores + w_val * val_scores
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the blended scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
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
[0] rationale: This metric isolates trials where the total number of positive features (the tally) is equal between Option A and Option B. On these 'tally tie' trials, Pure Tallying predicts exact indifference, resulting in an expected alignment with the higher-validity option of 50%. In contrast, Tallying with Validity Bias predicts that the secondary validity component will break the tie, leading to an alignment significantly greater than 50%.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([1.0, 0.9, 0.7, 0.5, 0.5])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    tally_a = a_mat.sum(axis=1)
    tally_b = b_mat.sum(axis=1)
    
    tie_mask = (tally_a == tally_b)
    if not np.any(tie_mask):
        return 0.5
        
    val_a = a_mat.dot(validities)
    val_b = b_mat.dot(validities)
    
    expected = np.where(val_a > val_b, 0, np.where(val_b > val_a, 1, -1))
    
    valid_mask = tie_mask & (expected != -1)
    if not np.any(valid_mask):
        return 0.5
        
    aligned = (data['response'].values[valid_mask] == expected[valid_mask])
    return float(np.mean(aligned))
outcome: self_sim=0.4961 (var=0.0063) adversary_sim=0.5344 (var=0.0073) welch_t=-1.644 p=0.1068 (N=25, alpha=0.01) -> reject

[1] rationale: Pure Tallying treats all features equally, so the probability of choosing the option with a +1 tally advantage should be identical regardless of the specific features. By subtracting the probability of choosing the tally-advantaged option on trials where the validity bias opposes the tally (Trials 3 & 4) from trials where the validity bias strongly supports the tally (Trials 5 & 8), Pure Tallying will predict a difference of exactly 0. Tallying with Validity Bias, however, will predict a robustly positive difference, as the validity bias increases the choice probability in the first set and decreases it in the second set. This within-subject contrast perfectly controls for the pure tallying effect while maximizing the validity difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    t5_mask = (a_tuples == (1, 1, 1, 0, 0)) & (b_tuples == (0, 0, 0, 1, 1))
    t8_mask = (a_tuples == (0, 0, 0, 1, 1)) & (b_tuples == (1, 1, 1, 0, 0))
    t3_mask = (a_tuples == (0, 0, 1, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    t4_mask = (a_tuples == (1, 1, 0, 0, 0)) & (b_tuples == (0, 0, 1, 1, 1))
    
    p_t5 = (data.loc[t5_mask, 'response'] == 0).mean() if t5_mask.sum() > 0 else 0.5
    p_t8 = (data.loc[t8_mask, 'response'] == 1).mean() if t8_mask.sum() > 0 else 0.5
    p_t3 = (data.loc[t3_mask, 'response'] == 0).mean() if t3_mask.sum() > 0 else 0.5
    p_t4 = (data.loc[t4_mask, 'response'] == 1).mean() if t4_mask.sum() > 0 else 0.5
    
    return float(((p_t5 + p_t8) / 2.0) - ((p_t3 + p_t4) / 2.0))
outcome: self_sim=0.0017 (var=0.0189) adversary_sim=0.0592 (var=0.0110) welch_t=-1.662 p=0.1035 (N=25, alpha=0.01) -> reject

[2] rationale: To robustly isolate the validity bias from the dominant tallying heuristic, this metric computes the within-subject residual choice probability after controlling for the exact tally difference. For every subject and every tally difference level (-1, 0, 1), we compute their empirical mean probability of choosing Option B. We then subtract this mean from their actual trial-by-trial choices to get a 'residual choice'. Under Pure Tallying, this residual is pure noise and is strictly uncorrelated with the difference in cue validities between the options. Under Tallying with Validity Bias, this residual will be positively correlated with the validity difference. By computing the covariance between the residual choice and the validity difference across all trials, we perfectly partial out the tallying effect, yielding an expected value of exactly 0 for Pure Tallying and a robustly positive value for the Competing Theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([1.0, 0.9, 0.7, 0.5, 0.5])
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    tally_diff = b_mat.sum(axis=1) - a_mat.sum(axis=1)
    val_diff = b_mat.dot(validities) - a_mat.dot(validities)
    
    df = pd.DataFrame({
        'td': tally_diff,
        'vd': val_diff,
        'resp': data['response'].values
    })
    
    if 'subject_id' in data.columns:
        df['subject_id'] = data['subject_id'].values
        td_means = df.groupby(['subject_id', 'td'])['resp'].transform('mean')
    else:
        td_means = df.groupby('td')['resp'].transform('mean')
        
    res = df['resp'] - td_means
    return float(np.mean(res * df['vd']))
outcome: self_sim=-0.0003 (var=0.0011) adversary_sim=0.0205 (var=0.0009) welch_t=-2.322 p=0.02452 (N=25, alpha=0.01) -> reject

[3] rationale: To maximize the discriminative signal while minimizing within-subject variance, this metric aggregates two independent choice contrasts that are both completely invariant under Pure Tallying. Term 1 evaluates 'tally tie' trials: under Pure Tallying, choices are random (expected value 0.5), but Tallying with Validity Bias predicts a preference for the option with higher cue validities. Term 2 evaluates 'tally difference' trials: it computes the probability of choosing the tally-favored option when the validity bias aligns with the tally minus the probability when it opposes the tally. Under Pure Tallying, this difference is exactly 0. By summing Term 1 (centered at 0) and Term 2, we pool the validity bias signals across all 96 trials, effectively doubling the mean difference between the theories while tightly constraining the variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    tally_a = a_mat.sum(axis=1)
    tally_b = b_mat.sum(axis=1)
    
    validities = np.array([1.0, 0.9, 0.7, 0.5, 0.5])
    val_a = a_mat.dot(validities)
    val_b = b_mat.dot(validities)
    
    resp = data['response'].values
    
    # Term 1: Tally ties
    tie_mask = (tally_a == tally_b)
    if np.any(tie_mask):
        val_a_tie = val_a[tie_mask]
        val_b_tie = val_b[tie_mask]
        resp_tie = resp[tie_mask]
        
        val_favored = np.where(val_a_tie > val_b_tie, 0, 1)
        chose_val_fav = (resp_tie == val_favored)
        term1 = np.mean(chose_val_fav) - 0.5
    else:
        term1 = 0.0
        
    # Term 2: Tally differences
    diff_mask = (tally_a != tally_b)
    if np.any(diff_mask):
        tally_a_diff = tally_a[diff_mask]
        tally_b_diff = tally_b[diff_mask]
        val_a_diff = val_a[diff_mask]
        val_b_diff = val_b[diff_mask]
        resp_diff = resp[diff_mask]
        
        chose_tally_fav = np.where(tally_a_diff > tally_b_diff, resp_diff == 0, resp_diff == 1)
        val_fav_diff = np.where(tally_a_diff > tally_b_diff, val_a_diff - val_b_diff, val_b_diff - val_a_diff)
        
        aligned_mask = val_fav_diff > 0
        opposed_mask = val_fav_diff < 0
        
        p_aligned = np.mean(chose_tally_fav[aligned_mask]) if np.any(aligned_mask) else 0.0
        p_opposed = np.mean(chose_tally_fav[opposed_mask]) if np.any(opposed_mask) else 0.0
        
        term2 = p_aligned - p_opposed
    else:
        term2 = 0.0
        
    return float(term1 + term2)
outcome: self_sim=0.0011 (var=0.0178) adversary_sim=0.0856 (var=0.0211) welch_t=-2.140 p=0.03747 (N=25, alpha=0.01) -> reject

[4] rationale: Pure Tallying predicts that when two options have the same number of positive features (a 'tally tie'), subjects will be exactly indifferent, regardless of the validities of those features. This metric isolates the two tally tie trials with the most extreme difference in cue validities. In Trial 2, Option B has a massive validity advantage (+0.85), whereas in Trial 1, Option B has a validity disadvantage (-0.40). By subtracting the probability of choosing Option B in Trial 1 from the probability of choosing Option B in Trial 2, Pure Tallying predicts a difference of exactly 0. In contrast, Tallying with Validity Bias predicts a robust positive difference, because the validity bias is most impactful when the tally is tied (avoiding the saturation of the softmax function that occurs on tally-difference trials). By focusing exclusively on tie trials, we maximize the discriminative signal while tightly controlling within-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 2: Tally tie where Option B has a large validity advantage (+0.85)
    t2_mask = (a_tuples == (0, 0, 0, 1, 1)) & (b_tuples == (1, 1, 0, 0, 0))
    
    # Trial 1: Tally tie where Option B has a validity disadvantage (-0.40)
    t1_mask = (a_tuples == (1, 0, 0, 0, 0)) & (b_tuples == (0, 0, 1, 0, 0))
    
    p_b_t2 = (data.loc[t2_mask, 'response'] == 1).mean() if t2_mask.sum() > 0 else 0.5
    p_b_t1 = (data.loc[t1_mask, 'response'] == 1).mean() if t1_mask.sum() > 0 else 0.5
    
    return float(p_b_t2 - p_b_t1)

outcome: self_sim=0.0138 (var=0.0216) adversary_sim=0.0338 (var=0.0350) welch_t=-0.420 p=0.6763 (N=25, alpha=0.01) -> reject

[5] rationale: Pure Tallying predicts that on trials where the two options have an equal number of positive features (Tally Ties), subjects will guess randomly (p=0.5), regardless of the cue validities. By restricting the analysis exclusively to Tally Ties, we eliminate all variance associated with the tallying heuristic and the unknown softmax temperature 'beta'. We compute the within-subject covariance between the choice and the validity difference on these tie trials. Under Pure Tallying, this covariance is exactly 0. Under Tallying with Validity Bias, choices will correlate positively with the validity difference, yielding a robustly positive metric. Mean-centering the responses per subject removes binomial noise from baseline guessing, strictly isolating the validity effect and maximizing the signal-to-noise ratio.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    # Extract feature matrices
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Compute tally difference (Option B - Option A)
    tally_diff = b_mat.sum(axis=1) - a_mat.sum(axis=1)
    
    # Compute validity difference using the exact experimental validities
    validities = np.array([0.95, 0.9, 0.55, 0.5, 0.5])
    val_diff = b_mat.dot(validities) - a_mat.dot(validities)
    
    # Create a working dataframe
    df = pd.DataFrame({
        'subject_id': data['subject_id'].values,
        'td': tally_diff,
        'vd': val_diff,
        'resp': data['response'].values
    })
    
    # Isolate Tally Ties (where pure Tallying predicts exactly 0.5 choice probability)
    df_ties = df[df['td'] == 0].copy()
    
    if len(df_ties) == 0:
        return 0.0
        
    # Mean-center responses within each subject to remove baseline guessing noise
    subj_means = df_ties.groupby('subject_id')['resp'].transform('mean')
    df_ties['resp_centered'] = df_ties['resp'] - subj_means
    
    # Compute the covariance between the centered response and the validity difference
    return float(np.mean(df_ties['resp_centered'] * df_ties['vd']))
outcome: self_sim=0.0033 (var=0.0010) adversary_sim=0.0106 (var=0.0017) welch_t=-0.711 p=0.4811 (N=25, alpha=0.01) -> reject

[6] rationale: This metric leverages the conflict trials to maximize the signal-to-noise ratio. Under Pure Tallying, the probability of choosing the tally-favored option depends strictly on the tally difference (diff=2 in Trial 6 vs diff=1 in Trials 3 and 4). Under Tallying with Validity Bias, the validity bias directly OPPOSES the tally advantage in Trials 3 and 4, severely shrinking the effective score difference and driving choice probabilities closer to 0.5. Conversely, in Trial 6, validity ALIGNS with the tally advantage, maintaining a high choice probability. By taking the difference in the probability of choosing the tally-favored option between Trial 6 and Trials 3/4, we capture the massive divergence in determinism caused by the validity penalty in conflict trials. This contrast yields a significantly larger mean difference between the theories than tie-trial metrics, overcoming the binomial variance boundary.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Convert ratings to numpy arrays
    a_mat = np.stack(data['option_a_ratings'].values)
    b_mat = np.stack(data['option_b_ratings'].values)
    
    # Calculate pure tally scores
    tally_a = a_mat.sum(axis=1)
    tally_b = b_mat.sum(axis=1)
    
    resp = data['response'].values
    
    # Identify Trial 6 (Tally difference = 2, Validity ALIGNS)
    # Trial 6: A=[0, 1, 0, 1, 1] (tally 3), B=[1, 0, 0, 0, 0] (tally 1)
    t6_mask = (tally_a == 3) & (tally_b == 1)
    
    # Identify Trials 3 & 4 (Tally difference = 1, Validity OPPOSES)
    # Trial 3: A=[0, 0, 1, 1, 1] (tally 3), B=[1, 1, 0, 0, 0] (tally 2)
    # Trial 4: A=[1, 1, 0, 0, 0] (tally 2), B=[0, 0, 1, 1, 1] (tally 3)
    t34_mask = ((tally_a == 3) & (tally_b == 2)) | ((tally_a == 2) & (tally_b == 3))
    
    # Calculate probability of choosing the tally-favored option
    p_tally_t6 = np.mean(resp[t6_mask] == 0) if np.any(t6_mask) else 0.5
    
    chose_tally_t34 = np.where(tally_a[t34_mask] > tally_b[t34_mask], resp[t34_mask] == 0, resp[t34_mask] == 1)
    p_tally_t34 = np.mean(chose_tally_t34) if np.any(t34_mask) else 0.5
    
    # Calculate determinism contrast
    # Under Pure Tallying, P(diff=2) is consistently higher than P(diff=1)
    # Under Validity Bias, w_val significantly reduces the score gap in Trials 3 & 4 (opposing),
    # creating a much larger spread between T6 and T3/4.
    return float(p_tally_t6 - p_tally_t34)
outcome: self_sim=0.0825 (var=0.0220) adversary_sim=0.1275 (var=0.0280) welch_t=-1.006 p=0.3197 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    validities = np.array([0.95, 0.9, 0.55, 0.5, 0.5])\n    \n    a_mat = np.stack(data['option_a_ratings'].values)\n    b_mat = np.stack(data['option_b_ratings'].values)\n    \n    val_a = a_mat.dot(validities)\n    val_b = b_mat.dot(validities)\n    \n    # Identify which option has the higher total validity\n    val_favored = np.where(val_b > val_a, 1, 0)\n    \n    # Calculate the overall proportion of choices that align with the higher-validity option\n    is_val_favored = (data['response'].values == val_favored)\n    \n    return float(np.mean(is_val_favored))",
  "rationale": "Instead of isolating specific trial types and throwing away data (which inflates within-subject variance), this metric computes the global proportion of trials where the subject chose the option with the higher objective validity. Under Pure Tallying, the expected value of this metric is tightly constrained because validity is completely ignored: on tally ties (Trials 1, 2, 5) it expects 50%, on conflict trials (Trials 3, 4) it expects a low probability (as tally opposes validity), and on aligned trials (Trial 6) it expects a high probability. Under Tallying with Validity Bias, the probability of choosing the higher-validity option strictly increases across all 6 trial types. By pooling all 96 trials, we capture this uniform positive shift while minimizing binomial noise, yielding a highly stable and discriminative metric."
}
```

## Usage

```json
{
  "prompt_token_count": 7924,
  "candidates_token_count": 425,
  "total_token_count": 11355
}
```
