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
**Validities (n_features=7):** [0.95, 0.85, 0.75, 0.65, 0.55, 0.51, 0.5]

**Trial pairs (n=9):**
  trial 1: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0, 0]
  trial 9: A=[0, 1, 1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Advocated theory (Take-The-Best with Compensatory Verification) from the Competing theory (Probabilistic Strategy Mixture) by exploiting the structural difference between score integration and probability mixing. In the Competing theory, the final choice probability is a mixture of the TTB and Tallying distributions. Consequently, if Option A wins the highest-validity cue, the probability of choosing Option A is strictly bounded from below by the probability of executing TTB; it can never fall below this asymptote, regardless of how many lower-validity cues Option B wins. In contrast, the Advocated theory integrates the TTB and Tallying evidence into a single score before the softmax choice rule. This compensatory mechanism allows a sufficiently large coalition of lower-validity cues for Option B to completely overwhelm Option A's TTB advantage, driving the probability of choosing Option A toward zero. By holding the TTB winner constant and systematically increasing the number of lower-validity cues Option B wins, we can observe whether the choice probabilities asymptote (supporting the Mixture model) or cross over and approach zero (supporting the Compensatory model).

**Computed schedule:** 9 unique pairs × 10 reps = 90 trials per subject.



## ADVOCATED THEORY
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


## COMPETING THEORY
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
[0] rationale: The Advocated theory uses a square-root tallying component, which predicts strong diminishing marginal returns for each additional lower-validity cue Option A wins. The difference in the probability of choosing A should be much larger when moving from 0 to 2 cues than when moving from 3 to 5 cues. The Competing theory's tallying component is linear in the log-odds space, which, combined with the mixture model, predicts a different shape of the choice probability curve. The metric directly quantifies the difference between the early jump and the late jump to capture this sub-linear signature.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    df = data.copy()
    df['A_sum'] = df['option_a_ratings'].apply(sum)
    df['A_has_0'] = df['option_a_ratings'].apply(lambda x: x[0])
    
    # Filter for trials 1-6 where Option B has the highest validity cue and A does not
    t1_6 = df[df['A_has_0'] == 0]
    
    # Calculate the probability of choosing A for each number of lower-validity cues A wins
    p_A = t1_6.groupby('A_sum')['response'].apply(lambda x: (x == 0).mean()).to_dict()
    
    def get_p(k):
        return p_A.get(k, 0.5)
        
    # Measure the non-linearity (diminishing marginal returns) 
    # by comparing the early jump (k=0 to 2) with the late jump (k=3 to 5)
    early_jump = get_p(2) - get_p(0)
    late_jump = get_p(5) - get_p(3)
    
    return float(early_jump - late_jump)
outcome: self_sim=0.2711 (var=0.1892) adversary_sim=0.3378 (var=0.1264) welch_t=-0.593 p=0.5559 (N=25, alpha=0.01) -> reject

[1] rationale: To reduce the high between-subject variance seen in previous difference-of-differences metrics, this metric aggregates across more trials while testing the core non-linear prediction of the Advocated theory. By computing the concavity of the choice probability curve (the difference between the mean choice probability of the intermediate values and the extremes), it expects a value near 0 for the point-symmetric logistic curve of the Competing theory, and a strictly positive value for the concave square-root tallying curve of the Advocated theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Identify trials 1-6 where Option A does not have the highest validity cue
    a_first = data['option_a_ratings'].apply(lambda x: x[0])
    df = data[a_first == 0].copy()
    
    if df.empty:
        return 0.0
        
    # Calculate the sum of cues won by A
    df['a_sum'] = df['option_a_ratings'].apply(sum)
    
    # Probability of choosing A
    df['choose_a'] = (df['response'] == 0).astype(float)
    
    # The Competing theory assumes a linear tallying process, which passed through 
    # a logistic function creates a point-symmetric curve. The average choice 
    # probability for the intermediate points (A_sum in 1,2,3,4) should equal 
    # the average of the extremes (A_sum in 0,5).
    # The Advocated theory uses a square root transformation for the tallying, 
    # creating a concave shape where the intermediate points sit higher than the 
    # linear interpolation between the extremes.
    
    mid_mask = df['a_sum'].isin([1, 2, 3, 4])
    end_mask = df['a_sum'].isin([0, 5])
    
    mid_mean = df.loc[mid_mask, 'choose_a'].mean()
    end_mean = df.loc[end_mask, 'choose_a'].mean()
    
    if pd.isna(mid_mean) or pd.isna(end_mean):
        return 0.0
        
    return float(mid_mean - end_mean)
outcome: self_sim=0.1222 (var=0.0327) adversary_sim=0.1250 (var=0.0194) welch_t=-0.061 p=0.9517 (N=25, alpha=0.01) -> reject

[2] rationale: The Competing theory relies on a linear tallying component passed through a logistic function. For Trials 1, 2, and 3, the tally differences are -1, 0, and +1, respectively. Because the logistic function is perfectly point-symmetric around 0, the increase in choice probability from Trial 1 to 2 is mathematically identical to the increase from Trial 2 to 3. Thus, the second difference (P3 - 2*P2 + P1) is exactly zero, regardless of the subject's fitted parameters. In contrast, the Advocated theory evaluates the remaining cues after the discriminating cue, yielding tally differences of 0, 1, and 2. These are compressed via a square root function to 0, 1.0, and 1.414. The sharp drop in marginal evidence (step size of 1.0 vs 0.414) creates a strong concavity, making the second difference consistently negative for almost all parameter combinations. This produces a stark mean difference between the theories while ensuring zero systematic between-subject variance for the Competing theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # The metric relies on the exact mathematical symmetry of the Competing theory
    # around the Tally_diff = 0 point. 
    # In trials 1, 2, and 3, Option B always wins the highest validity cue (TTB=B).
    # The Tally differences (Option A - Option B) for these trials are -1, 0, and +1.
    # For the Competing theory, the logistic function is point-symmetric around 0,
    # so the jump in choice probability from T1 to T2 exactly equals the jump from T2 to T3.
    # Therefore, P(A|T3) - 2*P(A|T2) + P(A|T1) is exactly 0 for the Competing theory.
    # 
    # For the Advocated theory, the tallying is compressed (square root).
    # The tally differences of the remaining cues are 0, 1, and 2.
    # Compressed, these are 0, 1.0, and 1.414.
    # The step size drops drastically (from 1.0 to 0.414), creating a strong concavity.
    # This makes the second difference P(A|T3) - 2*P(A|T2) + P(A|T1) strongly negative.
    
    # Identify trials 1, 2, and 3 based on Option A's sum of cues
    # T1: A_sum = 0. T2: A_sum = 1. T3: A_sum = 2.
    # All these trials have A's first cue == 0.
    
    df = data.copy()
    df['A_sum'] = df['option_a_ratings'].apply(sum)
    df['A_has_0'] = df['option_a_ratings'].apply(lambda x: x[0])
    
    # Filter for trials where A does not have the highest validity cue
    df_filtered = df[df['A_has_0'] == 0]
    
    if df_filtered.empty:
        return 0.0
        
    # Calculate mean response (probability of choosing A) for each A_sum
    p_A = df_filtered.groupby('A_sum')['response'].apply(lambda x: (x == 0).mean()).to_dict()
    
    p1 = p_A.get(0, 0.5)
    p2 = p_A.get(1, 0.5)
    p3 = p_A.get(2, 0.5)
    
    # Calculate the second difference
    return float(p3 - 2 * p2 + p1)

outcome: self_sim=-0.1378 (var=0.1623) adversary_sim=0.0133 (var=0.0793) welch_t=-1.537 p=0.1316 (N=25, alpha=0.01) -> reject

[3] rationale: The Competing theory assumes a linear strategy mixture, meaning the TTB and Tally probabilities are computed independently and then averaged. Because the mixture is additive in probability space, the effect of increasing Option A's Tally sum is completely independent of which option won the TTB cue. Thus, the increase in choice probability when Option A gains cues (from sum=1 to sum>1) should be mathematically identical whether A or B won the first cue, yielding an expected double-difference of exactly 0. The Advocated theory, however, integrates TTB and Tally evidence inside a single logistic function and uses sub-linear (square root) tallying on the remaining cues. This creates a strong interaction: the marginal impact of additional tally cues depends heavily on whether they are compensating for a lost TTB cue or adding to a won TTB cue, making this metric systematically positive for the Advocated theory.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    df = data.copy()
    df['A_sum'] = df['option_a_ratings'].apply(sum)
    df['A_first'] = df['option_a_ratings'].apply(lambda x: x[0])
    
    # Calculate P(A) for each condition
    p_a = df.groupby(['A_first', 'A_sum'])['response'].apply(lambda x: (x == 0).mean()).to_dict()
    
    # Effect A: A wins TTB (A_first == 1)
    # T7: A_sum == 1
    # T8, T9, T10: A_sum == 2, 3, 4
    p_t7 = p_a.get((1, 1), 0.5)
    p_t8_10 = sum(p_a.get((1, s), 0.5) for s in [2, 3, 4]) / 3.0
    effect_a = p_t8_10 - p_t7
    
    # Effect B: B wins TTB (A_first == 0)
    # T2: A_sum == 1
    # T3, T4, T5: A_sum == 2, 3, 4
    p_t2 = p_a.get((0, 1), 0.5)
    p_t3_5 = sum(p_a.get((0, s), 0.5) for s in [2, 3, 4]) / 3.0
    effect_b = p_t3_5 - p_t2
    
    return float(effect_a - effect_b)
outcome: self_sim=0.0874 (var=0.1335) adversary_sim=-0.0326 (var=0.0491) welch_t=+1.404 p=0.1681 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    df = data.copy()\n    df['a_has_0'] = df['option_a_ratings'].apply(lambda x: x[0])\n    df['a_sum'] = df['option_a_ratings'].apply(sum)\n    df['b_sum'] = df['option_b_ratings'].apply(sum)\n    \n    # T1: A wins TTB, B has 1 tally cue.\n    t1_mask = (df['a_has_0'] == 1) & (df['b_sum'] == 1)\n    p_ttb_t1 = (df.loc[t1_mask, 'response'] == 0).mean() if t1_mask.any() else 0.5\n    \n    # T5, T6: A wins TTB, B has 5 or 6 tally cues.\n    t56_mask = (df['a_has_0'] == 1) & (df['b_sum'].isin([5, 6]))\n    p_ttb_t56 = (df.loc[t56_mask, 'response'] == 0).mean() if t56_mask.any() else 0.5\n    \n    # T7: B wins TTB, A has 1 tally cue.\n    t7_mask = (df['a_has_0'] == 0) & (df['a_sum'] == 1)\n    p_ttb_t7 = (df.loc[t7_mask, 'response'] == 1).mean() if t7_mask.any() else 0.5\n    \n    # T9: B wins TTB, A has 6 tally cues.\n    t9_mask = (df['a_has_0'] == 0) & (df['a_sum'] == 6)\n    p_ttb_t9 = (df.loc[t9_mask, 'response'] == 1).mean() if t9_mask.any() else 0.5\n    \n    # Calculate the drop in the probability of choosing the TTB winner \n    # when the opposing option gains a massive tally advantage.\n    drop_A = p_ttb_t1 - p_ttb_t56\n    drop_B = p_ttb_t7 - p_ttb_t9\n    \n    return float((drop_A + drop_B) / 2.0)",
  "rationale": "The Mixture model assumes the final choice probability is a linear combination of the TTB and Tallying probabilities. When the TTB winner is held constant and the other option gains tally cues, the Tallying probability for the TTB winner drops from 0.5 to 0. Therefore, the maximum possible drop in the overall choice probability is exactly 0.5 * (1 - p_ttb). Since p_ttb is bounded between 0 and 1, this drop is strictly bounded and averages around 0.25 across subjects. The Compensatory model, in contrast, integrates all evidence into a single score before passing it through the logistic function. A sufficiently large tally advantage for the opposing option can completely overwhelm the TTB advantage, driving the choice probability from near 1.0 all the way down to near 0.0. This predicts a much larger probability drop, averaging around 0.6. By measuring the drop in the probability of choosing the TTB winner when the opposing option goes from 1 to 5/6 tally cues, we can reliably discriminate the bounded asymptote of the Mixture model from the full crossover of the Compensatory model."
}
```

## Usage

```json
{
  "prompt_token_count": 6432,
  "candidates_token_count": 855,
  "total_token_count": 25178
}
```
