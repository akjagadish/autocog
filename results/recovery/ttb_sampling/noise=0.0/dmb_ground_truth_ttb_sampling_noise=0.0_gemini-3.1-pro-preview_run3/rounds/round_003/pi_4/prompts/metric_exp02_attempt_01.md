# metric_exp02_attempt_01

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
**Validities (n_features=5):** [0.92, 0.88, 0.84, 0.8, 0.76]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 7: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]

**Rationale:** To strongly dissociate Probabilistic Take-The-Best (PTTB) from deterministic Take-The-Best (TTB), we exploit their fundamentally different ways of resolving conflicts among discriminating features. TTB strictly chooses the option favored by the single highest-validity discriminating feature, entirely ignoring the rest. PTTB, however, forms a probabilistic choice based on an exponentially weighted sum of all discriminating features. We use a 5-feature design with closely spaced validities. In critical trials, Option A is favored by the single most valid discriminating feature, while Option B is favored by multiple slightly less valid features. TTB deterministically predicts Option A in these cases. In contrast, PTTB will produce graded choice probabilities that can shift toward Option B (especially for lower or moderate values of the gamma parameter), because the combined exponential weights of B's multiple features can exceed the weight of A's single feature. We also include trials where the most valid features are tied to test the fallback mechanisms of both models.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Take-The-Best (PTTB) with exponential weighting: Decision-makers search through features sequentially but stochastically. The probability of checking a feature is proportional to an exponential function of its validity (exp(gamma * validity)), allowing high-validity features to strongly dominate the search order. The search stops at the first feature that discriminates between the options, and the option favored by that feature is chosen. If no features discriminate, the decision-maker guesses. This yields a Luce choice rule over the exponentially weighted discriminating features.

**Parameters:**
- gamma: [0.1, 200.0]
- epsilon: [0.0, 0.25]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("PTTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Identify discriminating features for each option
    d_a = (a > b).astype(float)
    d_b = (b > a).astype(float)
    d_any = d_a + d_b
    
    if np.sum(d_any) == 0:
        p_a = 0.5
    else:
        # To prevent numerical overflow, subtract max validity among discriminating features
        max_v = np.max(validities[d_any > 0])
        weights = np.exp(gamma * (validities - max_v))
        
        # Sum of weights of features favoring A and B
        w_a = np.sum(d_a * weights)
        w_b = np.sum(d_b * weights)
        
        # The probability that the first discriminating feature found favors A
        p_a = w_a / (w_a + w_b)
        
    p_core = np.array([p_a, 1.0 - p_a])
    
    # Incorporate uniform lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Take-The-Best (TTB) heuristic: People make decisions by considering features one at a time, ordered by their validity. They stop at the first feature that discriminates between the options and choose the option with the higher value on that feature. If no features discriminate, they guess. This is a non-compensatory, lexicographic process.

**Parameters:**
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    validities = np.asarray(parameters["validities"], dtype=float)
    epsilon = float(parameters["epsilon"])
    
    # Sort features by validity descending
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
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
[0] rationale: This metric isolates 'conflict' trials where the deterministic Take-The-Best (TTB) choice (driven by the single highest validity discriminating feature) disagrees with the simple Tallying choice (the option with the greater total number of positive features). TTB will almost never choose the Tallying option on these trials (rate near 0, driven only by the lapse rate epsilon). Probabilistic Take-The-Best (PTTB), however, samples features proportionally to an exponential weighting of their validities. Since the conflicting option often has several slightly lower-validity features, PTTB will choose the Tallying option at a noticeably higher rate, sharply distinguishing the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.85, 0.7, 0.9, 0.75, 0.8])
    conflict_tally_choices = 0
    conflict_trials = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        sum_a = np.sum(a)
        sum_b = np.sum(b)
        d_a = (a > b).astype(float)
        d_b = (b > a).astype(float)
        if np.sum(d_a) + np.sum(d_b) > 0:
            max_v_a = np.max(validities * d_a) if np.any(d_a) else -1
            max_v_b = np.max(validities * d_b) if np.any(d_b) else -1
            ttb_choice = 0 if max_v_a > max_v_b else 1
            tally_choice = 0 if sum_a > sum_b else (1 if sum_b > sum_a else -1)
            if tally_choice != -1 and ttb_choice != tally_choice:
                conflict_trials += 1
                if row['response'] == tally_choice:
                    conflict_tally_choices += 1
    return conflict_tally_choices / conflict_trials if conflict_trials > 0 else 0.0
outcome: self_sim=0.1131 (var=0.0163) adversary_sim=0.1231 (var=0.0082) welch_t=-0.319 p=0.7512 (N=25, alpha=0.01) -> reject

[1] rationale: This metric computes the within-subject difference in accuracy (defined as choosing the TTB-predicted option) between 'easy' trials and 'hard' trials. 'Easy' trials are those where the non-TTB option has zero discriminating features (e.g., trials 10 and 11), meaning both theories unequivocally favor the TTB option up to the lapse rate. 'Hard' trials are those where the non-TTB option is favored by multiple features whose maximum validity is only marginally lower (diff <= 0.05) than the TTB option's best feature. For the deterministic TTB theory, the probability of choosing the TTB winner is strictly driven by the lapse rate and is identical across easy and hard trials, yielding an expected difference of 0. For Probabilistic TTB, the presence of competing features on hard trials draws probability mass away from the TTB winner, while easy trials do not. Thus, PTTB will exhibit a strictly positive difference, reliably discriminating the two models regardless of the specific parameters.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.85, 0.7, 0.9, 0.75, 0.8])
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        easy_correct = 0
        easy_total = 0
        hard_correct = 0
        hard_total = 0
        
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            
            d_a = (a > b).astype(float)
            d_b = (b > a).astype(float)
            
            max_v_a = np.max(validities * d_a) if np.any(d_a > 0) else -1
            max_v_b = np.max(validities * d_b) if np.any(d_b > 0) else -1
            
            if max_v_a > max_v_b:
                ttb_winner = 0
                diff = max_v_a - max_v_b if max_v_b >= 0 else 999
            elif max_v_b > max_v_a:
                ttb_winner = 1
                diff = max_v_b - max_v_a if max_v_a >= 0 else 999
            else:
                continue
                
            is_correct = (row['response'] == ttb_winner)
            
            if diff > 1.0:
                easy_total += 1
                if is_correct:
                    easy_correct += 1
            elif diff <= 0.06:
                hard_total += 1
                if is_correct:
                    hard_correct += 1
                    
        if easy_total > 0 and hard_total > 0:
            easy_acc = easy_correct / easy_total
            hard_acc = hard_correct / hard_total
            diffs.append(easy_acc - hard_acc)
            
    if not diffs:
        return 0.0
    return float(np.mean(diffs))
outcome: self_sim=0.0564 (var=0.0217) adversary_sim=0.0013 (var=0.0080) welch_t=+1.601 p=0.1173 (N=25, alpha=0.01) -> reject

[2] rationale: Due to the parameter distributions, PTTB is largely indistinguishable from TTB structurally for the vast majority of subjects (since gamma is drawn up to 200, effectively making the exponential weights heavily favor the highest validity feature). However, the two theories have significantly different assumptions about the lapse rate (epsilon). TTB assumes a wider range of lapse rates (epsilon up to 0.5), while PTTB assumes a tighter range (epsilon up to 0.25). By computing the overall accuracy with respect to the deterministic TTB prediction across all trials, we can reliably discriminate the two models: TTB will have a lower mean accuracy and higher between-subject variance due to its larger epsilon range, whereas PTTB will consistently show higher accuracy.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    validities = np.array([0.85, 0.7, 0.9, 0.75, 0.8])
    match_count = 0
    total = 0
    for _, row in data.iterrows():
        a = np.array(row['option_a_ratings'])
        b = np.array(row['option_b_ratings'])
        d_a = (a > b).astype(float)
        d_b = (b > a).astype(float)
        max_v_a = np.max(validities * d_a) if np.any(d_a > 0) else -1
        max_v_b = np.max(validities * d_b) if np.any(d_b > 0) else -1
        if max_v_a > max_v_b:
            ttb_choice = 0
        elif max_v_b > max_v_a:
            ttb_choice = 1
        else:
            continue
        if row['response'] == ttb_choice:
            match_count += 1
        total += 1
    return float(match_count / total) if total > 0 else 0.5

outcome: self_sim=0.8983 (var=0.0108) adversary_sim=0.8781 (var=0.0079) welch_t=+0.737 p=0.4646 (N=25, alpha=0.01) -> reject

[3] rationale: This metric perfectly isolates the difference in the lapse rate (epsilon) distributions between the two theories by exclusively analyzing 'easy' trials (trials 10 and 11). On these trials, one option strictly dominates the other (zero discriminating features for the worse option). Consequently, the Probabilistic Take-The-Best (PTTB) model's gamma parameter has no effect, as the weight of the dominated option is zero. Thus, on easy trials, both PTTB and TTB predict the dominant option with probability 1 - epsilon/2.

Because PTTB restricts epsilon to [0.0, 0.25] while TTB allows epsilon up to 0.5, TTB produces a heavy upper tail of errors on these trials that PTTB cannot produce. Specifically, with 16 easy trials per subject, PTTB subjects will almost never make more than 2 errors (only ~9.5% of the time due to binomial noise), whereas TTB subjects will make more than 2 errors ~50% of the time. By computing the proportion of subjects who make strictly more than 2 errors on easy trials, we construct a metric with a massive mean difference and strictly bounded variance, completely sidestepping the high variance induced by the PTTB gamma parameter and ensuring reliable discrimination at N=25.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    subj_metrics = []
    for subj, subj_df in data.groupby('subject_id'):
        errors = 0
        easy_trials = 0
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            d_a = (a > b).astype(float)
            d_b = (b > a).astype(float)
            
            sum_a = np.sum(d_a)
            sum_b = np.sum(d_b)
            
            if sum_a == 0 and sum_b == 0:
                continue
                
            if sum_a == 0 or sum_b == 0:
                easy_trials += 1
                correct_choice = 0 if sum_a > 0 else 1
                if row['response'] != correct_choice:
                    errors += 1
                    
        if easy_trials > 0:
            # > 0.1251 means strictly more than 2 errors out of 16 trials
            if errors / easy_trials > 0.1251:
                subj_metrics.append(1.0)
            else:
                subj_metrics.append(0.0)
                
    if not subj_metrics:
        return 0.0
        
    return float(np.mean(subj_metrics))
outcome: self_sim=0.1000 (var=0.0900) adversary_sim=0.3600 (var=0.2304) welch_t=-2.297 p=0.02692 (N=25, alpha=0.01) -> reject

[4] rationale: This metric computes the within-subject difference in the probability of choosing the deterministic Take-The-Best (TTB) predicted option between two sets of trials: 'Low Conflict' and 'High Conflict'. 'High Conflict' trials are those where the non-TTB option has at least 2 MORE discriminating features than the TTB option (e.g., Trial 1 and 3). 'Low Conflict' trials are those where the non-TTB option has equal or fewer discriminating features (e.g., Trials 6, 7, 8). 

Because deterministic TTB only considers the single highest validity feature and strictly ignores all others, its probability of choosing the TTB-predicted option is completely insensitive to the number of opposing features. Consequently, for a TTB agent, the expected difference in choice rates between these two sets of trials is exactly 0, with very tight variance governed only by binomial noise. 

Conversely, Probabilistic TTB (PTTB) forms a choice probability based on the sum of exponentially weighted validities. In 'High Conflict' trials, the large number of opposing features draws significant probability mass away from the TTB winner, whereas in 'Low Conflict' trials, the TTB winner retains most of the mass. As a result, PTTB produces a strictly positive difference. By contrasting an expectation of exactly 0 (TTB) against a positive shift (PTTB), we achieve a highly reliable discriminative metric with minimal between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    validities = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        w_low = 0
        total_low = 0
        w_high = 0
        total_high = 0
        
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            
            d_a = (a > b).astype(float)
            d_b = (b > a).astype(float)
            
            max_v_a = np.max(validities * d_a) if np.any(d_a > 0) else -1
            max_v_b = np.max(validities * d_b) if np.any(d_b > 0) else -1
            
            if max_v_a > max_v_b:
                w_choice = 0
                num_w = np.sum(d_a)
                num_l = np.sum(d_b)
            elif max_v_b > max_v_a:
                w_choice = 1
                num_w = np.sum(d_b)
                num_l = np.sum(d_a)
            else:
                continue
                
            is_w_chosen = (row['response'] == w_choice)
            
            diff_features = num_l - num_w
            
            if diff_features >= 2:
                total_high += 1
                if is_w_chosen:
                    w_high += 1
            elif diff_features <= 0:
                total_low += 1
                if is_w_chosen:
                    w_low += 1
                    
        if total_low > 0 and total_high > 0:
            rate_low = w_low / total_low
            rate_high = w_high / total_high
            diffs.append(rate_low - rate_high)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=0.0306 (var=0.0116) adversary_sim=0.0089 (var=0.0077) welch_t=+0.780 p=0.4393 (N=25, alpha=0.01) -> reject

[5] rationale: This metric computes the within-subject difference in the probability of choosing the non-TTB option between two sets of trials: 'Small Diff' (validity difference <= 0.06) and 'Large Diff' (validity difference > 0.06). For the deterministic TTB theory, the probability of choosing the non-TTB option is strictly driven by the lapse rate (epsilon) and is identical across all trials, yielding an expected difference of exactly 0 with extremely low variance. For Probabilistic TTB, the choice probability depends on the exponentially weighted validities; trials with a smaller difference between the best features result in a significantly higher probability of choosing the non-TTB option than trials with a larger difference. Thus, PTTB will exhibit a strictly positive difference, reliably discriminating the two models with minimal between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    validities = np.array([0.9, 0.85, 0.8, 0.75, 0.7])
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        non_ttb_small = 0
        total_small = 0
        non_ttb_large = 0
        total_large = 0
        
        for _, row in subj_df.iterrows():
            a = np.array(row['option_a_ratings'])
            b = np.array(row['option_b_ratings'])
            
            d_a = (a > b).astype(float)
            d_b = (b > a).astype(float)
            
            max_v_a = np.max(validities * d_a) if np.any(d_a > 0) else -1
            max_v_b = np.max(validities * d_b) if np.any(d_b > 0) else -1
            
            if max_v_a > max_v_b:
                diff = max_v_a - max_v_b if max_v_b >= 0 else 999
                is_non_ttb = (row['response'] == 1)
            elif max_v_b > max_v_a:
                diff = max_v_b - max_v_a if max_v_a >= 0 else 999
                is_non_ttb = (row['response'] == 0)
            else:
                continue
                
            if diff <= 0.06:
                total_small += 1
                if is_non_ttb:
                    non_ttb_small += 1
            elif diff > 0.06 and diff < 1.0:
                total_large += 1
                if is_non_ttb:
                    non_ttb_large += 1
                    
        if total_small > 0 and total_large > 0:
            rate_small = non_ttb_small / total_small
            rate_large = non_ttb_large / total_large
            diffs.append(rate_small - rate_large)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=0.0461 (var=0.0109) adversary_sim=-0.0131 (var=0.0064) welch_t=+2.252 p=0.02928 (N=25, alpha=0.01) -> reject

[6] rationale: In this experimental design, the deterministic Take-The-Best (TTB) model always predicts Option A across all 8 trial types. Thus, for a TTB agent, any choices for Option B are purely the result of the lapse rate (epsilon). Consequently, the true probability of choosing Option B is identical across all trial types. The variance of the empirical B-choice rates across the 8 trial types will strictly equal the expected binomial noise (p(1-p)/12), yielding an expected 'excess variance' of exactly 0.

In contrast, the Probabilistic Take-The-Best (PTTB) model computes choice probabilities based on the sum of exponentially weighted validities. Because the trial types vary significantly in the number and validity of features supporting Option B, the true probability of choosing B varies across trial types (especially for subjects with a low gamma parameter). Therefore, the empirical variance of B-choice rates across trial types will significantly exceed the binomial noise, yielding a strictly positive 'excess variance'. By subtracting the expected binomial noise from the sample variance, we isolate the true variance in choice probabilities, creating a metric that centers tightly at 0 for TTB and is strictly positive for PTTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    excess_variances = []
    for subj, subj_df in data.groupby('subject_id'):
        # Group by trial type (using option strings as unique identifiers)
        subj_df = subj_df.copy()
        subj_df['trial_key'] = subj_df.apply(lambda row: str(row['option_a_ratings']) + str(row['option_b_ratings']), axis=1)
        
        # Calculate B-choice rate for each trial type
        trial_rates = subj_df.groupby('trial_key')['response'].mean().values
        
        # Overall B-choice rate for the subject
        overall_rate = subj_df['response'].mean()
        
        # Sample variance of the 8 trial-type rates
        # using ddof=1 for unbiased sample variance
        if len(trial_rates) > 1:
            s2 = np.var(trial_rates, ddof=1)
        else:
            s2 = 0.0
            
        # Expected binomial variance if all trials had the same true probability
        # Since there are 12 repetitions per trial type:
        expected_binomial_var = (overall_rate * (1.0 - overall_rate)) / 12.0
        
        # Excess variance
        excess_var = s2 - expected_binomial_var
        excess_variances.append(excess_var)
        
    if not excess_variances:
        return 0.0
        
    return float(np.mean(excess_variances))
outcome: self_sim=0.0031 (var=0.0000) adversary_sim=0.0002 (var=0.0000) welch_t=+1.862 p=0.06955 (N=25, alpha=0.01) -> reject

[7] rationale: In this specific experimental design, the deterministic Take-The-Best (TTB) model predicts Option A as the winner across all 8 trial types. Consequently, for a pure TTB agent, any choices for Option B are solely the result of the random lapse rate (epsilon). This means the true probability of choosing Option B is perfectly identical across all trials. By splitting the 8 trial types into two equal halves (Set 1 and Set 2) and computing the within-subject difference in B-choice rates, TTB will yield an expected difference of exactly 0, with between-subject variance minimized to pure binomial noise over a large number of trials (48 per set).

Conversely, Probabilistic Take-The-Best (PTTB) forms choice probabilities via an exponential weighting of all discriminating features. Set 1 is specifically chosen to contain trials where Option B has many valid features that closely trail Option A's best feature, drawing significant probability mass toward B. Set 2 contains trials where Option B's features are either fewer or substantially lower in validity, resulting in a lower probability of choosing B. Therefore, PTTB will produce a strictly positive difference, robustly distinguishing it from TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    import pandas as pd
    
    set1_pairs = {
        ((1,0,0,0,0), (0,1,1,1,1)),
        ((1,0,0,1,1), (0,1,1,1,1)),
        ((1,1,0,0,0), (1,0,1,1,1)),
        ((1,0,1,0,0), (0,1,0,1,1))
    }
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        set1_b_choices = 0
        set1_total = 0
        set2_b_choices = 0
        set2_total = 0
        
        for _, row in subj_df.iterrows():
            a_tuple = tuple(row['option_a_ratings'])
            b_tuple = tuple(row['option_b_ratings'])
            
            is_b = (row['response'] == 1)
            
            if (a_tuple, b_tuple) in set1_pairs:
                set1_total += 1
                if is_b:
                    set1_b_choices += 1
            else:
                set2_total += 1
                if is_b:
                    set2_b_choices += 1
                    
        if set1_total > 0 and set2_total > 0:
            rate1 = set1_b_choices / set1_total
            rate2 = set2_b_choices / set2_total
            diffs.append(rate1 - rate2)
            
    if not diffs:
        return 0.0
        
    return float(np.mean(diffs))
outcome: self_sim=0.0200 (var=0.0069) adversary_sim=-0.0021 (var=0.0037) welch_t=+1.072 p=0.2896 (N=25, alpha=0.01) -> reject

[8] rationale: This metric contrasts the rate of choosing Option B between two extreme sets of trials. In both Set 1 and Set 2, deterministic Take-The-Best (TTB) predicts Option A, meaning any choices for Option B are purely due to the lapse rate (epsilon). Thus, for TTB, the expected difference in the B-choice rate between these sets is exactly 0. In contrast, Probabilistic Take-The-Best (PTTB) evaluates the exponentially weighted sum of all validities. Set 1 features trials where Option B has many discriminating features, which pulls probability mass toward B. Set 2 features trials where Option A dominates in both highest validity and total number of features. Therefore, PTTB will produce a strictly positive difference, robustly distinguishing the two theories with very low between-subject variance.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Set 1: Trials where Option B has many more features than Option A
    set1_pairs = {
        ((1, 0, 0, 0, 0), (0, 1, 1, 1, 1)),
        ((0, 1, 0, 0, 0), (0, 0, 1, 1, 1))
    }
    # Set 2: Trials where Option A has more features than Option B
    set2_pairs = {
        ((1, 0, 1, 0, 1), (0, 1, 0, 1, 0)),
        ((1, 1, 1, 0, 0), (0, 0, 0, 1, 1))
    }
    
    diffs = []
    for subj, subj_df in data.groupby('subject_id'):
        s1_b = 0
        s1_tot = 0
        s2_b = 0
        s2_tot = 0
        
        for _, row in subj_df.iterrows():
            a = tuple(row['option_a_ratings'])
            b = tuple(row['option_b_ratings'])
            
            if (a, b) in set1_pairs:
                s1_tot += 1
                s1_b += row['response']
            elif (a, b) in set2_pairs:
                s2_tot += 1
                s2_b += row['response']
                
        if s1_tot > 0 and s2_tot > 0:
            diffs.append((s1_b / s1_tot) - (s2_b / s2_tot))
            
    if not diffs:
        return 0.0
    return float(np.mean(diffs))
outcome: self_sim=0.0489 (var=0.0242) adversary_sim=-0.0256 (var=0.0106) welch_t=+1.995 p=0.05256 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Set 1: Trials where the non-TTB option has many more discriminating features.\n    # This strongly pulls PTTB's choice probability away from the TTB option.\n    set1_pairs = {\n        ((1, 0, 0, 0, 0), (0, 1, 1, 1, 1)), # Trial 1: TTB chooses A\n        ((0, 1, 0, 0, 0), (0, 0, 1, 1, 1)), # Trial 4: TTB chooses A\n        ((0, 0, 1, 0, 0), (0, 0, 0, 1, 1))  # Trial 7: TTB chooses A\n    }\n    \n    # Set 2: Trials where the TTB option also has more discriminating features.\n    # This strongly reinforces PTTB's probability of choosing the TTB option.\n    set2_pairs = {\n        ((1, 1, 1, 0, 0), (0, 0, 0, 1, 1)), # Trial 9: TTB chooses A\n        ((1, 0, 1, 0, 0), (1, 1, 0, 1, 1))  # Trial 5: TTB chooses B\n    }\n    \n    diffs = []\n    for subj, subj_df in data.groupby('subject_id'):\n        s1_ttb_choices = 0\n        s1_tot = 0\n        s2_ttb_choices = 0\n        s2_tot = 0\n        \n        for _, row in subj_df.iterrows():\n            a = tuple(row['option_a_ratings'])\n            b = tuple(row['option_b_ratings'])\n            \n            if (a, b) in set1_pairs:\n                s1_tot += 1\n                # For all Set 1 trials, TTB predicts Option A (response == 0)\n                if row['response'] == 0:\n                    s1_ttb_choices += 1\n            elif (a, b) in set2_pairs:\n                s2_tot += 1\n                # For Trial 9, TTB predicts Option A. For Trial 5, TTB predicts Option B.\n                if a == (1, 1, 1, 0, 0):  # Trial 9\n                    if row['response'] == 0:\n                        s2_ttb_choices += 1\n                else:  # Trial 5\n                    if row['response'] == 1:\n                        s2_ttb_choices += 1\n                        \n        if s1_tot > 0 and s2_tot > 0:\n            rate1 = s1_ttb_choices / s1_tot\n            rate2 = s2_ttb_choices / s2_tot\n            diffs.append(rate2 - rate1)\n            \n    if not diffs:\n        return 0.0\n        \n    return float(np.mean(diffs))",
  "rationale": "This metric isolates the effect of conflicting feature counts on choice probabilities. We define Set 1 as trials where the deterministic Take-The-Best (TTB) option is heavily outnumbered in discriminating features by the non-TTB option. We define Set 2 as trials where the TTB option is supported by a greater number of discriminating features than the non-TTB option. For a deterministic TTB agent, the probability of choosing the TTB option is driven exclusively by a constant lapse rate (epsilon) across all trials, meaning the expected difference in TTB choice rates between Set 2 and Set 1 is exactly 0, with variance limited to pure binomial noise. Conversely, Probabilistic Take-The-Best (PTTB) samples features proportionally to their exponentially weighted validities. In Set 1, the multitude of opposing features draws significant probability mass away from the TTB option. In Set 2, the additional supporting features reinforce the TTB option. Thus, PTTB will exhibit a strictly positive difference. By contrasting an expectation of exactly 0 (TTB) against a robust positive shift (PTTB) using carefully balanced trial subsets, we achieve highly reliable discrimination with minimal between-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 10076,
  "candidates_token_count": 1030,
  "total_token_count": 40199
}
```
