# metric_exp00_attempt_02

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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 5: A=[0, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 6: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 7: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 0, 0, 0]
  trial 8: A=[0, 0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1, 0]

**Rationale:** This design quantitatively dissociates the Probabilistic Strategy Mixture (Advocated) from Take-The-Best with Compensatory Verification (Competing) by exploiting the Competing model's sub-linear (square root) transformation of the tally difference. In the Advocated model, Tallying is strictly linear: the log-odds contribution of the Tallying component grows linearly with the absolute difference in the total number of winning cues. In the Competing model, the cues evaluated after the discriminating cue are tallied and then compressed via a square root function before being integrated with the TTB score. By creating a series of trials where Option A wins the highest-validity cue and an exponentially increasing number of lower-validity cues (e.g., +0, +1, +2, +4 advantages in remaining cues), the Advocated model predicts a constant, linear increase in the log-odds of choosing Option A for each additional cue. In contrast, the Competing model predicts a strongly diminishing marginal increase in log-odds due to its sub-linear compression. We also include reversed trials to fully map the symmetry and rule out alternative models.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Strategy Mixture (Soft TTB and Soft Tallying)

**Parameters:**
- p_ttb: [0.0, 1.0]
- beta_ttb: [0.1, 10.0]
- beta_tally: [0.1, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # 1. Take-The-Best (TTB) Prediction (Probabilistic)
    ttb_winner = 0.5
    for j in cue_order:
        if a[j] > b[j]:
            ttb_winner = 1.0
            break
        elif b[j] > a[j]:
            ttb_winner = 0.0
            break
            
    ttb_scores = np.array([ttb_winner, 1.0 - ttb_winner])
    beta_ttb = float(parameters["beta_ttb"])
    z_ttb = beta_ttb * ttb_scores
    z_ttb = z_ttb - np.max(z_ttb)
    p_ttb_dist = np.exp(z_ttb) / np.sum(np.exp(z_ttb))
    
    # 2. Tallying Prediction
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    
    beta_tally = float(parameters["beta_tally"])
    z_tally = beta_tally * np.array([tally_a, tally_b])
    z_tally = z_tally - np.max(z_tally)
    p_tally_dist = np.exp(z_tally) / np.sum(np.exp(z_tally))
    
    # 3. Strategy Mixture
    p_ttb = float(parameters["p_ttb"])
    p_core = p_ttb * p_ttb_dist + (1.0 - p_ttb) * p_tally_dist
    
    return p_core
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
**Description:** Take-The-Best with Compensatory Verification (Sub-linear Tallying)

**Parameters:**
- beta_ttb: [0.0, 10.0]
- beta_tally: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    ttb_winner = -1
    disc_idx = -1
    for i, j in enumerate(cue_order):
        if a[j] != b[j]:
            ttb_winner = 0 if a[j] > b[j] else 1
            disc_idx = i
            break
            
    if ttb_winner == -1:
        return np.array([0.5, 0.5])
        
    # Tally remaining cues (those evaluated after the discriminating cue)
    remaining_a = 0.0
    remaining_b = 0.0
    for j in cue_order[disc_idx+1:]:
        remaining_a += a[j]
        remaining_b += b[j]
        
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    
    tally_diff = remaining_a - remaining_b
    compressed_diff = np.sign(tally_diff) * np.sqrt(np.abs(tally_diff))
    
    score_a = beta_ttb * (1.0 if ttb_winner == 0 else 0.0) + beta_tally * compressed_diff
    score_b = beta_ttb * (1.0 if ttb_winner == 1 else 0.0)
    
    scores = np.array([score_a, score_b])
    z = scores - np.max(scores)
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
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
[0] rationale: This metric calculates the second difference of the probability of choosing the Take-The-Best (TTB) winner as a function of the number of extra supporting cues. The Advocated model uses a linear Tallying component, meaning the probability jump from 0 to 2 extra cues should be similar to the jump from 2 to 4 extra cues (making the second difference ~0). The Competing model uses a square-root compression for remaining cues, meaning the marginal effect of extra cues strongly diminishes, making the jump from 2 to 4 extra cues much smaller than from 0 to 2. Thus, the metric will be significantly more negative for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    def get_ttb_winner_and_extra(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        if a[0] > b[0]:
            ttb_winner = 0
            extra = sum(a[1:]) - sum(b[1:])
        else:
            ttb_winner = 1
            extra = sum(b[1:]) - sum(a[1:])
        return pd.Series({'ttb_winner': ttb_winner, 'extra': extra})

    res = data.apply(get_ttb_winner_and_extra, axis=1)
    data_eval = data.copy()
    data_eval['ttb_winner'] = res['ttb_winner']
    data_eval['extra'] = res['extra']
    data_eval['chose_ttb'] = (data_eval['response'] == data_eval['ttb_winner']).astype(float)
    
    p_diff0 = data_eval[data_eval['extra'] == 0]['chose_ttb'].mean()
    p_diff2 = data_eval[data_eval['extra'] == 2]['chose_ttb'].mean()
    p_diff4 = data_eval[data_eval['extra'] == 4]['chose_ttb'].mean()
    
    if pd.isna(p_diff0): p_diff0 = 0.5
    if pd.isna(p_diff2): p_diff2 = 0.5
    if pd.isna(p_diff4): p_diff4 = 0.5
    
    return float((p_diff4 - p_diff2) - (p_diff2 - p_diff0))
outcome: self_sim=-0.0275 (var=0.0168) adversary_sim=-0.0775 (var=0.0355) welch_t=+1.093 p=0.2806 (N=25, alpha=0.01) -> reject

[1] rationale: This metric calculates the empirical log-odds of choosing the TTB winner for trials with 0, 1, and 4 extra supporting cues, and returns L4 - 2*L1 + L0. In the Competing model, the sub-linear tallying compresses 0, 1, and 4 extra cues into score differences of 0, 1, and 2, making the log-odds increase perfectly linear with respect to the compressed values. Thus, L4 - 2*L1 + L0 should be approximately 0. In the Advocated model, the linear tallying component produces score differences of 1, 2, and 5 (since the first cue is also tallied), meaning the jump from 0 to 4 extra cues is much larger than twice the jump from 0 to 1, leading to a strongly positive metric value.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np

    def get_ttb_winner_and_extra(row):
        a = row['option_a_ratings']
        b = row['option_b_ratings']
        if a[0] > b[0]:
            ttb_winner = 0
            extra = sum(a[1:]) - sum(b[1:])
        else:
            ttb_winner = 1
            extra = sum(b[1:]) - sum(a[1:])
        return pd.Series({'ttb_winner': ttb_winner, 'extra': extra})

    res = data.apply(get_ttb_winner_and_extra, axis=1)
    data_eval = data.copy()
    data_eval['ttb_winner'] = res['ttb_winner']
    data_eval['extra'] = res['extra']
    data_eval['chose_ttb'] = (data_eval['response'] == data_eval['ttb_winner']).astype(int)
    
    def calc_logit(df_sub, extra_val):
        subset = df_sub[df_sub['extra'] == extra_val]
        if len(subset) == 0:
            return 0.0
        n_ttb = subset['chose_ttb'].sum()
        n_total = len(subset)
        return np.log((n_ttb + 0.5) / (n_total - n_ttb + 0.5))

    L0 = calc_logit(data_eval, 0)
    L1 = calc_logit(data_eval, 1)
    L4 = calc_logit(data_eval, 4)
    
    return float(L4 - 2 * L1 + L0)
outcome: self_sim=-0.3336 (var=1.0781) adversary_sim=-0.2950 (var=2.2624) welch_t=-0.106 p=0.9165 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    def is_ttb_choice(row):\n        # Cue 0 is the highest validity cue (0.95)\n        a_wins_ttb = row['option_a_ratings'][0] > row['option_b_ratings'][0]\n        ttb_winner = 0 if a_wins_ttb else 1\n        return 1.0 if row['response'] == ttb_winner else 0.0\n        \n    return float(data.apply(is_ttb_choice, axis=1).mean())",
  "rationale": "This metric calculates the overall proportion of trials where the subject chooses the option favored by the highest-validity cue (the Take-The-Best winner). The Advocated theory (Probabilistic Strategy Mixture) has no lapse rate parameter and its Tallying component includes the highest-validity cue, meaning both strategies consistently point towards the TTB winner, leading to a very high overall alignment (~0.95+). In contrast, the Competing theory (TTB with Compensatory Verification) includes an epsilon parameter (lapse rate up to 0.5) and its Tallying component excludes the highest-validity cue, providing zero initial tally support on baseline trials. Consequently, the Competing model predicts a significantly lower overall probability of choosing the TTB winner (~0.84). This structural difference in baseline support and lapsing creates a massive, robust mean gap with low between-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 4906,
  "candidates_token_count": 350,
  "total_token_count": 32447
}
```
