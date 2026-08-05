# metric_exp01_attempt_00

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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.55, 0.52, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 1]
  trial 3: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 4: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 1, 1, 0]  B=[1, 0, 1, 0, 0, 0]
  trial 6: A=[0, 0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 1, 1]  B=[0, 1, 0, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Rationale:** This design dissociates pure Tallying from Thresholded Tallying by exploiting a sharp difference in cue validities. The 6 features are split into a high-validity group (cues 1-3) and a low-validity group (cues 4-6). Pure Tallying ignores these validities and simply counts the total number of positive features. Thresholded Tallying, however, applies a soft filter that down-weights or eliminates the low-validity cues. By pitting options with many low-validity endorsements against options with fewer high-validity endorsements, pure Tallying will choose the option with the higher total count, whereas Thresholded Tallying will choose the option with the higher count among the high-validity cues. Ties in Tallying are also broken by Thresholded Tallying based on the distribution of high-validity cues.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by tallying the total number of positive features for each option, ignoring cue validities entirely. The option with the higher unweighted sum of positive features is chosen. This Equal Weight (or Tallying) heuristic provides a frugal but fully compensatory strategy, capturing the strong human tendency to prefer options with multiple supporting cues over those with a single high-validity cue. Response noise is modeled via a softmax over the tallied scores with inverse temperature beta, and an independent lapse rate epsilon.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
        
    # Tallying: count the number of positive features (unweighted sum) for each option.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution.
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Thresholded Tallying (Soft Subset Tallying): Decision makers use an equal-weight heuristic, but first filter the available information by cue validity. Instead of a hard threshold, they use a soft, probabilistic inclusion criterion where features with validity near or above a subjective threshold are more likely to be considered. These softly filtered features are then tallied to form a preference. This differentiable thresholding allows smooth transitions between pure Tallying (all cues included) and stricter subset tallying, reflecting boundedly rational cue selection that is easier to optimize during learning or fitting.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- theta: [-1.0, 1.0]
- k: [1.0, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Thresholded Tallying expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    theta = float(parameters["theta"])
    k = float(parameters["k"])
    
    # Filter features: soft threshold using a logistic sigmoid
    mask = 1.0 / (1.0 + np.exp(-k * (validities - theta)))
    
    # Tallying: count the number of positive features among the softly kept ones
    scores = np.sum(stim * mask, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    n_opts = p_core.shape[0]
    # Blend with uniform lapse distribution
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities
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
[0] rationale: This metric calculates the proportion of choices consistent with the pure Tallying heuristic specifically on 'conflict' trials. Conflict trials are defined as those where one option has a higher total number of positive features, but the other option has a higher number of highly valid features (the top 2 features). Tallying predicts subjects will consistently choose the option with more total features, yielding a metric value close to 1. Thresholded Tallying (which filters out low-validity features) predicts subjects will choose the option with more high-validity features, yielding a metric value closer to 0.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    sum_a = a_ratings.sum(axis=1)
    sum_b = b_ratings.sum(axis=1)
    
    top2_a = a_ratings[:, :2].sum(axis=1)
    top2_b = b_ratings[:, :2].sum(axis=1)
    
    # Identify conflict trials where the option with more total features has fewer top-2 features
    conflict_mask = ((sum_a > sum_b) & (top2_a < top2_b)) | ((sum_a < sum_b) & (top2_a > top2_b))
    
    if not np.any(conflict_mask):
        return 0.5
        
    responses = data['response'].values
    
    # Check if the chosen option is the one with the higher total number of features (Tallying)
    chose_tallying = np.zeros(len(data), dtype=bool)
    chose_tallying[(sum_a > sum_b) & (responses == 0)] = True
    chose_tallying[(sum_a < sum_b) & (responses == 1)] = True
    
    return float(np.mean(chose_tallying[conflict_mask]))
outcome: self_sim=0.8440 (var=0.0094) adversary_sim=0.7450 (var=0.0625) welch_t=+1.847 p=0.07434 (N=25, alpha=0.01) -> reject

[1] rationale: Pure Tallying treats Trial 1, Trial 2, and Trial 7 identically: in all three trials, one option has exactly 1 more positive feature than the other. Thus, the probability of choosing the option with more features should be identical across these trials, making the expected difference 0 (with very low variance). Thresholded Tallying, however, drops low-validity features. In Trial 1, the option with more total features has 2 fewer high-validity features, whereas in Trials 2 and 7 it only has 1 fewer high-validity feature. Consequently, Thresholded Tallying penalizes the 'more features' option much more severely in Trial 1, resulting in a systematically negative metric.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_tuples = data['option_a_ratings'].apply(tuple)
    b_tuples = data['option_b_ratings'].apply(tuple)
    
    # Trial 1: Tallying choice has +1 total features, but -2 top-validity features
    t1_fwd = (a_tuples == (0,0,1,1,1)) & (b_tuples == (1,1,0,0,0))
    t1_rev = (a_tuples == (1,1,0,0,0)) & (b_tuples == (0,0,1,1,1))
    
    # Trial 2: Tallying choice has +1 total features, and -1 top-validity features
    t2_fwd = (a_tuples == (0,0,1,1,0)) & (b_tuples == (1,0,0,0,0))
    t2_rev = (a_tuples == (1,0,0,0,0)) & (b_tuples == (0,0,1,1,0))
    
    # Trial 7: Tallying choice has +1 total features, and -1 top-validity features
    t7_fwd = (a_tuples == (0,0,0,1,1)) & (b_tuples == (0,1,0,0,0))
    t7_rev = (a_tuples == (0,1,0,0,0)) & (b_tuples == (0,0,0,1,1))
    
    def get_prob(mask_fwd, mask_rev):
        fwd_choices = (data.loc[mask_fwd, 'response'] == 0).sum()
        rev_choices = (data.loc[mask_rev, 'response'] == 1).sum()
        total = mask_fwd.sum() + mask_rev.sum()
        if total == 0:
            return 0.5
        return (fwd_choices + rev_choices) / total

    p1 = get_prob(t1_fwd, t1_rev)
    p2 = get_prob(t2_fwd, t2_rev)
    p7 = get_prob(t7_fwd, t7_rev)
    
    # Metric: P(Tallying choice in Trial 1) - average P(Tallying choice in Trial 2 & 7)
    return float(p1 - 0.5 * (p2 + p7))
outcome: self_sim=-0.0067 (var=0.0199) adversary_sim=-0.0233 (var=0.0193) welch_t=+0.421 p=0.6757 (N=25, alpha=0.01) -> reject

[2] rationale: By splitting the features into the 'Top 2' (high validity) and 'Bottom 3' (low validity), we can measure how often subjects follow the majority of top features (W_top) versus the majority of bottom features (W_bot). In this specific experimental design, the trial pairs are constructed such that whenever Top and Bottom features conflict, the Bottom features always have a larger difference in count (e.g., a difference of 3 vs a difference of 2). Because Tallying strictly counts total features, it will consistently side with the Bottom features in these conflicts, leading to W_bot ~ 1.0 and W_top ~ 0.0 (Metric ~ -1.0). Thresholded Tallying, however, actively down-weights or drops the bottom features, causing it to side with the Top features much more often, pulling the metric significantly higher (closer to 0.0 or even positive).
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    responses = data['response'].values
    
    # Top 2 features
    top_a = a_ratings[:, :2].sum(axis=1)
    top_b = b_ratings[:, :2].sum(axis=1)
    
    # Bottom 3 features
    bot_a = a_ratings[:, 2:].sum(axis=1)
    bot_b = b_ratings[:, 2:].sum(axis=1)
    
    # W_top: fraction of times choice agrees with Top features
    top_differs = top_a != top_b
    if not np.any(top_differs):
        w_top = 0.5
    else:
        top_agrees = ((top_a > top_b) & (responses == 0)) | ((top_a < top_b) & (responses == 1))
        w_top = float(np.mean(top_agrees[top_differs]))
        
    # W_bot: fraction of times choice agrees with Bottom features
    bot_differs = bot_a != bot_b
    if not np.any(bot_differs):
        w_bot = 0.5
    else:
        bot_agrees = ((bot_a > bot_b) & (responses == 0)) | ((bot_a < bot_b) & (responses == 1))
        w_bot = float(np.mean(bot_agrees[bot_differs]))
        
    return float(w_top - w_bot)
outcome: self_sim=-0.6919 (var=0.0337) adversary_sim=-0.5140 (var=0.2164) welch_t=-1.778 p=0.08517 (N=25, alpha=0.01) -> reject

[3] rationale: By computing the Pearson correlation between the subject's binary choices and the score difference derived solely from the Top-2 features, we exploit the adversarial structure of the trial pairs. In this design, the total feature sum (used by Tallying) is inherently negatively correlated with the Top-2 feature sum. Consequently, pure Tallying produces choices that are consistently and robustly negatively correlated with the Top-2 difference. Thresholded Tallying, which often filters out the bottom 3 features, relies heavily on the Top-2 features, systematically shifting this correlation to be much less negative or even positive. Correlation is bounded and normalizes out subject-specific noise (like inverse temperature beta), ensuring tight within-theory variance while maximizing the between-theory gap.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # We compute the Pearson correlation between the subject's choice 
    # and the score difference of ONLY the Top-2 (highest validity) features.
    # 
    # The experimental design is adversarial: the option with MORE total features 
    # frequently has FEWER top-2 features. As a result, the pure Tallying score 
    # difference is strongly NEGATIVELY correlated with the Top-2 score difference.
    # 
    # Tallying subjects strictly follow the total feature count, so their choices 
    # will exhibit a strong NEGATIVE correlation with the Top-2 difference.
    # Thresholded Tallying subjects probabilistically drop low-validity features, 
    # causing their choices to align much more often with the Top-2 difference, 
    # pulling this correlation significantly higher (closer to 0 or positive).
    
    # Convert response (0 for A, 1 for B) to a binary indicator for choosing A
    y = 1.0 - data['response'].values
    
    a_ratings = np.stack(data['option_a_ratings'].values)
    b_ratings = np.stack(data['option_b_ratings'].values)
    
    # Extract the sum of the Top-2 highest validity features for each option
    top2_a = a_ratings[:, :2].sum(axis=1)
    top2_b = b_ratings[:, :2].sum(axis=1)
    
    # Difference in Top-2 features (Score A - Score B)
    z = top2_a - top2_b
    
    # Return the Pearson correlation coefficient
    if np.std(y) == 0 or np.std(z) == 0:
        return 0.0
        
    return float(np.corrcoef(y, z)[0, 1])
outcome: self_sim=-0.5124 (var=0.0210) adversary_sim=-0.3586 (var=0.1420) welch_t=-1.904 p=0.06623 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    \n    sum_a = a_ratings.sum(axis=1)\n    sum_b = b_ratings.sum(axis=1)\n    \n    top3_a = a_ratings[:, :3].sum(axis=1)\n    top3_b = b_ratings[:, :3].sum(axis=1)\n    \n    # Isolate trials where total features are tied but top-3 features differ\n    valid_mask = (sum_a == sum_b) & (top3_a != top3_b)\n    \n    if not np.any(valid_mask):\n        return 0.5\n        \n    responses = data['response'].values\n    \n    # Did the subject choose the option with more top-3 features?\n    chose_high_val = np.zeros(len(data), dtype=bool)\n    chose_high_val[(top3_a > top3_b) & (responses == 0)] = True\n    chose_high_val[(top3_a < top3_b) & (responses == 1)] = True\n    \n    return float(np.mean(chose_high_val[valid_mask]))",
  "rationale": "Pure Tallying strictly counts the total number of positive features. On trials where both options have the exact same number of positive features (Trials 2 and 4), Tallying is completely indifferent and will choose either option with 50% probability, regardless of the subject's noise parameters. Thresholded Tallying, however, applies a soft filter that down-weights lower-validity features. Even when its threshold is low and it mimics Tallying, the logistic filter strictly preserves the order of validities, meaning it will always assign a higher score to the option with more high-validity features. By measuring the proportion of choices for the option with more high-validity features specifically on trials where the total feature counts are tied, we isolate a condition where Tallying predicts exactly 0.5 (with minimal variance) while Thresholded Tallying predicts a value strictly greater than 0.5."
}
```

## Usage

```json
{
  "prompt_token_count": 6024,
  "candidates_token_count": 551,
  "total_token_count": 19030
}
```
