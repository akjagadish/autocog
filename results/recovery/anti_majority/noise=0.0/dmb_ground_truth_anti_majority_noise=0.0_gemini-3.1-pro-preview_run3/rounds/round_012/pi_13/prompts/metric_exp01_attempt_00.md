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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=7):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 3: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 6: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 0, 0]
  trial 7: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Advocated model (WADD-PWR) from the Competing model (Take-The-Best with Compensatory Verification) by exploiting the Competing model's strict tallying of remaining cues. In the Competing model, once the highest-validity discriminating cue establishes the 'TTB winner', all subsequent cues are merely tallied, completely ignoring their specific validities. By creating a sequence of trials where the highest-validity discriminating cue and the net difference in the remaining winning cues are held strictly constant, the Competing model is mathematically forced to predict identical choice probabilities. However, by shifting which specific lower-validity cues belong to Option A versus Option B across these trials, we alter the underlying validity distribution. Consequently, the Advocated WADD-PWR model, which integrates all cues using a power function of their specific validities, predicts systematically shifting choice probabilities across trials where the Competing model predicts strict equivalence.

**Computed schedule:** 7 unique pairs × 13 reps = 91 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive with Power-Function Validity Scaling (WADD-PWR): Decision-makers integrate all available cues but weight them non-linearly using a power function of their stated validities. A single parameter, gamma, controls the weighting scheme: positive values approximate Take-The-Best by heavily favoring the most valid cues, zero yields Tallying by weighting all cues equally, and negative values approximate Reverse TTB by favoring less valid cues. The power function naturally anchors a validity of 1.0 to a weight of 1.0, providing built-in numerical stability across the full spectrum of strategies without requiring artificial normalization.

**Parameters:**
- gamma: [-20.0, 20.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Apply power-function weighting scheme to capture TTB, Tallying, and Reverse TTB
    # Validities are in [0.5, 1.0], so val ** gamma is numerically stable.
    w = val ** gamma
    
    # Accumulate evidence based on discriminating cues
    diff = a - b
    ev_a = np.sum(w[diff > 0])
    ev_b = np.sum(w[diff < 0])
    
    # Convert to choice probabilities via softmax
    scores = np.array([ev_a, ev_b])
    z = beta * scores
    e = np.exp(z - np.max(z))
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
[0] rationale: This metric evaluates the curvature of choice probability changes as a function of the number of remaining winning cues. The Advocated model assumes linear evidence accumulation, so the step from 1 to 4 remaining cues produces a log-odds change three times larger than the step from 0 to 1 cue. The Competing model assumes a square-root transformation, making these log-odds steps equal. By calculating the difference in probability jumps ((P3 - P2) - (P2 - P1) and similarly for the negative side), we capture this structural divergence in a simple, stable scalar that avoids infinite log-odds.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    
    p_A = (data['response'] == 0).astype(float)
    
    p1 = p_A[(b_sum == 0) & (a_sum == 1)].mean()
    p2 = p_A[(b_sum == 0) & (a_sum == 2)].mean()
    p3 = p_A[(b_sum == 0) & (a_sum == 5)].mean()
    
    p4 = p_A[(a_sum == 1) & (b_sum == 1)].mean()
    p5 = p_A[(a_sum == 1) & (b_sum == 4)].mean()
    
    curv_pos = (p3 - p2) - (p2 - p1)
    curv_neg = (p4 - p5) - (p1 - p4)
    
    return float(curv_pos + curv_neg)
outcome: self_sim=-0.3937 (var=0.3368) adversary_sim=-0.1800 (var=0.3332) welch_t=-1.306 p=0.1979 (N=25, alpha=0.01) -> reject

[1] rationale: By analyzing the log-odds of choosing Option A, we can linearly separate the structural predictions of the two models. The Competing model transforms the sum of remaining cues sub-linearly (using a square root), predicting that the log-odds step from 1 to 4 remaining cues equals the step from 0 to 1. This implies L3 - L2 ≈ L2 - L1, and symmetrically L1 - L4 ≈ L4 - L5. By calculating L3 - 2*L2 + 2*L4 - L5, the Competing model evaluates to approximately 0. In contrast, the Advocated model assumes linear evidence accumulation, meaning the step from 1 to 4 cues is three times larger than from 0 to 1, causing this metric to deviate significantly from 0. Using smoothed empirical log-odds stabilizes the variance per subject.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    
    t2 = (a_sum == 2) & (b_sum == 0)
    t3 = (a_sum == 5) & (b_sum == 0)
    t4 = (a_sum == 1) & (b_sum == 1)
    t5 = (a_sum == 1) & (b_sum == 4)
    
    def log_odds(mask):
        subset = data[mask]
        if len(subset) == 0:
            return 0.0
        count_A = (subset['response'] == 0).sum()
        count_B = len(subset) - count_A
        return float(np.log((count_A + 0.5) / (count_B + 0.5)))
        
    L2 = log_odds(t2)
    L3 = log_odds(t3)
    L4 = log_odds(t4)
    L5 = log_odds(t5)
    
    return float(L3 - 2 * L2 + 2 * L4 - L5)
outcome: self_sim=-2.2186 (var=13.3724) adversary_sim=-1.3041 (var=11.3995) welch_t=-0.919 p=0.3629 (N=25, alpha=0.01) -> reject

[2] rationale: Instead of relying on log-odds which can explode and induce huge between-subject variances, this metric applies a linear contrast directly to the choice probabilities. By comparing the spread in choice probability when the tally difference is +/- 4 against when it is +/- 1, we can robustly separate the linear evidence accumulation of the Advocated model from the sub-linear (square-root) accumulation of the Competing model. The coefficient 2.5 acts as a threshold that separates the theoretical ratios (approx 2 for Competing vs approx 4 for Advocated), ensuring the two models fall on opposite sides of 0 or at least far apart.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    
    p_A = (data['response'] == 0).astype(float)
    
    p2 = p_A[(a_sum == 2) & (b_sum == 0)].mean()
    p3 = p_A[(a_sum == 5) & (b_sum == 0)].mean()
    p4 = p_A[(a_sum == 1) & (b_sum == 1)].mean()
    p5 = p_A[(a_sum == 1) & (b_sum == 4)].mean()
    
    # p3 and p5 represent a net remaining difference of +4 and -4 respectively.
    # p2 and p4 represent a net remaining difference of +1 and -1 respectively.
    # Under the Competing model's square-root tallying, the log-odds difference between p3 and p5
    # is 2x the difference between p2 and p4.
    # Under the Advocated model's linear tallying, the difference is 4x.
    # We test (p3 - p5) - 2.5 * (p2 - p4) directly on probabilities to avoid log-odds blowups.
    return float((p3 - p5) - 2.5 * (p2 - p4))

outcome: self_sim=-0.6319 (var=0.5739) adversary_sim=-0.3806 (var=0.5348) welch_t=-1.193 p=0.2387 (N=25, alpha=0.01) -> reject

[3] rationale: This metric isolates the trials where the most valid cue (cue 1) is in direct opposition to four lower-validity cues. In Trial 5, Option A wins cue 1 while Option B wins four cues. In Trial 6, this is perfectly reversed. By computing the difference in the probability of choosing Option A between Trial 6 and Trial 5 (p6 - p5), we measure the net influence of the four lower-validity cues over the single highest-validity cue. The Advocated model (linear accumulation) predicts these four cues will have a much stronger combined weight (4x the weight of one lower cue) compared to the Competing model (square-root tallying, which predicts only 2x the weight). Therefore, p6 - p5 will be significantly more positive for the Advocated model, providing a highly stable, direct behavioral contrast without the variance inflation of log-odds or complex ratios.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    a_sum = data['option_a_ratings'].apply(sum)
    b_sum = data['option_b_ratings'].apply(sum)
    
    p_A = (data['response'] == 0).astype(float)
    
    # Trial 5: A wins cue 1, B wins cues 2,3,4,5. A_sum = 1, B_sum = 4
    t5_mask = (a_sum == 1) & (b_sum == 4)
    p5 = p_A[t5_mask].mean() if t5_mask.sum() > 0 else 0.5
    
    # Trial 6: B wins cue 1, A wins cues 2,3,4,5. A_sum = 4, B_sum = 1
    t6_mask = (a_sum == 4) & (b_sum == 1)
    p6 = p_A[t6_mask].mean() if t6_mask.sum() > 0 else 0.5
    
    return float(p6 - p5)

outcome: self_sim=0.3625 (var=0.4673) adversary_sim=0.4662 (var=0.4174) welch_t=-0.552 p=0.5838 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    data['A_tuple'] = data['option_a_ratings'].apply(tuple)\n    data['B_tuple'] = data['option_b_ratings'].apply(tuple)\n    \n    is_A = (data['response'] == 0).astype(float)\n    \n    t1 = (data['A_tuple'] == (1, 1, 0, 0, 0)) & (data['B_tuple'] == (0, 0, 1, 1, 0))\n    t4 = (data['A_tuple'] == (1, 0, 0, 0, 1)) & (data['B_tuple'] == (0, 1, 1, 0, 0))\n    t5 = (data['A_tuple'] == (1, 1, 1, 0, 0)) & (data['B_tuple'] == (0, 0, 0, 1, 0))\n    t7 = (data['A_tuple'] == (1, 0, 1, 1, 0)) & (data['B_tuple'] == (0, 1, 0, 0, 0))\n    \n    p1 = is_A[t1].mean() if t1.sum() > 0 else 0.5\n    p4 = is_A[t4].mean() if t4.sum() > 0 else 0.5\n    p5 = is_A[t5].mean() if t5.sum() > 0 else 0.5\n    p7 = is_A[t7].mean() if t7.sum() > 0 else 0.5\n    \n    return float((p1 - p4) + (p5 - p7))",
  "rationale": "By design, the Competing model identifies Cue 1 as the Take-The-Best discriminator in all trials and simply tallies the remaining cues. In trials 1 and 4, the net difference in the remaining tally is exactly -1 for Option A. In trials 5 and 7, the net difference is exactly +1. Consequently, the Competing model is mathematically forced to predict identical choice probabilities for trials 1 and 4, and identically for trials 5 and 7, yielding a metric expectation of 0. The Advocated model, however, integrates the specific validities of all cues. Option A has much stronger lower-validity cues in trial 1 compared to trial 4, and in trial 5 compared to trial 7. Thus, the Advocated model predicts a strongly positive divergence, robustly separating the theories."
}
```

## Usage

```json
{
  "prompt_token_count": 5539,
  "candidates_token_count": 629,
  "total_token_count": 9354
}
```
