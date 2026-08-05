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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 6: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 7: A=[0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates the advocated Sequential Evidence Accumulation (SEA) theory from the competing Environment-Contingent Strategy Selection theory by exploiting the difference between true sequential early-stopping and a fixed mixture of heuristics. The competing theory proposes a constant probabilistic mixture of Take-The-Best (TTB) and Tallying for any given environment (since dispersion is fixed per experiment). Consequently, it predicts that choice probabilities will always shift when the raw tally difference changes, because the Tallying component is always active to some degree. The advocated SEA theory, however, accumulates evidence sequentially and halts as soon as a threshold is met. If the first cue provides enough evidence to cross the threshold, SEA completely ignores all subsequent cues, predicting absolutely identical choice probabilities across trials with different tally differences as long as the top cue is unchanged. By presenting trials where the top cue strongly favors Option A but the remaining cues systematically shift the tally advantage toward Option B, we can cleanly test whether lower-validity cues are partially integrated (competing theory's Tallying component) or completely ignored due to early stopping (advocated theory).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Sequential Evidence Accumulation: Decision-makers evaluate cues sequentially in descending order of validity. Each cue provides evidence proportional to a non-linear transformation of its validity above chance. Evidence is accumulated as a running difference between the two options. If the absolute accumulated evidence exceeds a threshold, search is terminated and a choice is made based on the current evidence. If all cues are evaluated without crossing the threshold, a decision is made based on the final accumulated evidence. This allows for fast, non-compensatory decisions when top cues are highly valid, while gracefully falling back to compensatory integration when early cues are less decisive.

**Parameters:**
- theta: [0.0, 10.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale weights by transforming validity above chance, allowing better separation
    weights = np.maximum(val - 0.5, 0.001) ** gamma
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if absolute accumulated evidence reaches the threshold
            if abs(E) >= theta:
                break
            
    scores = np.array([E, -E])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probabilities = np.asarray(probs, dtype=np.float64)
    probabilities /= probabilities.sum()
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Environment-Contingent Strategy Selection with Mean-Relative Dominant Cue Sensitivity: Decision-makers select between non-compensatory (Take-The-Best) and compensatory (Tallying) heuristics based on the structural properties of the environment. Specifically, the probability of deploying Take-The-Best increases as a logistic function of the difference between the top cue's validity and the average validity of all cues. In environments where the top cue strongly stands out from the overall cue distribution, individuals rely on TTB; when validities are relatively flat, they fall back to Tallying.

**Parameters:**
- gamma: [0.1, 50.0]
- threshold: [0.0, 0.5]
- beta_ttb: [0.1, 20.0]
- beta_tally: [0.1, 20.0]
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
    threshold = float(parameters["threshold"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_tally = float(parameters["beta_tally"])
    epsilon = float(parameters["epsilon"])
    
    # Structural property: top cue vs mean validity
    sorted_val = np.sort(val)[::-1]
    dispersion = sorted_val[0] - np.mean(val)
    
    # Probability of using TTB based on environment structure
    # Using logistic function to map dispersion to probability
    p_ttb_strategy = 1.0 / (1.0 + np.exp(-gamma * (dispersion - threshold)))
    
    # Take-The-Best (TTB) Strategy
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        elif b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.0, 0.0])
        
    z_ttb = beta_ttb * scores_ttb
    z_ttb = z_ttb - np.max(z_ttb)
    e_ttb = np.exp(z_ttb)
    p_ttb_choice = e_ttb / np.sum(e_ttb)
    
    # Tallying Strategy
    tally_a = np.sum(a)
    tally_b = np.sum(b)
    scores_tally = np.array([tally_a, tally_b])
    
    z_tally = beta_tally * scores_tally
    z_tally = z_tally - np.max(z_tally)
    e_tally = np.exp(z_tally)
    p_tally_choice = e_tally / np.sum(e_tally)
    
    # Mixture of strategies
    p_mix = p_ttb_strategy * p_ttb_choice + (1.0 - p_ttb_strategy) * p_tally_choice
    
    # Lapse rate
    p_final = (1.0 - epsilon) * p_mix + epsilon * np.array([0.5, 0.5])
    
    return p_final
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
[0] rationale: Measures the difference in the probability of choosing Option A between Trial 3 (where Tallying favors B by 1) and Trial 1 (where Tallying favors B by 3). Under the competing ECSS theory, the constant mixture of Tallying causes a significant drop in P(A) for Trial 1 relative to Trial 3. Under the advocated SEA theory, early stopping on the highly valid first cue leads to identical choices for both trials, resulting in a difference near zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1_mask = (a_str == '10000') & (b_str == '01111')
    t3_mask = (a_str == '10000') & (b_str == '00011')
    
    p_a_t1 = (data.loc[t1_mask, 'response'] == 0).mean()
    p_a_t3 = (data.loc[t3_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t1): p_a_t1 = 0.0
    if pd.isna(p_a_t3): p_a_t3 = 0.0
    
    return float(p_a_t3 - p_a_t1)
outcome: self_sim=0.1933 (var=0.1154) adversary_sim=0.0217 (var=0.0236) welch_t=+2.303 p=0.02767 (N=25, alpha=0.01) -> reject

[1] rationale: Under the competing ECSS theory, the probability of using Take-The-Best is constant across all trials. In both Trial 4 and Trial 8, the highest valid discriminating cue favors Option A, meaning the TTB component always chooses A. Additionally, the Tally difference in both trials is exactly -1 (Option B has one more positive cue than Option A), meaning the Tallying component yields identical choice probabilities for both trials. Consequently, ECSS predicts P(A|T4) and P(A|T8) to be identical, making their difference exactly zero in expectation. Under the advocated SEA theory, however, Trial 4 has the two most valid cues favoring Option A, while Trial 8 only has the third most valid cue favoring Option A. SEA will accumulate much more evidence for A in Trial 4 than in Trial 8, predicting a substantially higher P(A) for Trial 4. This metric isolates this clean dissociation.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t4_mask = (a_str == '11000') & (b_str == '00111')
    t8_mask = (a_str == '00100') & (b_str == '00011')
    
    p_a_t4 = (data.loc[t4_mask, 'response'] == 0).mean()
    p_a_t8 = (data.loc[t8_mask, 'response'] == 0).mean()
    
    if pd.isna(p_a_t4): p_a_t4 = 0.5
    if pd.isna(p_a_t8): p_a_t8 = 0.5
    
    return float(p_a_t4 - p_a_t8)
outcome: self_sim=0.0783 (var=0.0518) adversary_sim=-0.0083 (var=0.0253) welch_t=+1.560 p=0.126 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_str = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    b_str = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    t4_mask = (a_str == '11000') & (b_str == '00111')\n    t5_mask = (a_str == '11100') & (b_str == '00011')\n    t6_mask = (a_str == '01000') & (b_str == '00111')\n    t7_mask = (a_str == '01000') & (b_str == '00011')\n    \n    p_a = {}\n    for i, mask in enumerate([t4_mask, t5_mask, t6_mask, t7_mask]):\n        if mask.sum() > 0:\n            p_a[i] = (data.loc[mask, 'response'] == 0).mean()\n        else:\n            p_a[i] = 0.5\n            \n    return float((p_a[1] - p_a[0]) + (p_a[3] - p_a[2]))",
  "rationale": "Measures the effect of altering lower-validity cues while keeping the top discriminating cues constant. Under the competing ECSS theory, the Tallying component is always active to some degree. For Trials 5 vs 4, the Tally advantage shifts from +1 to -1 for Option A, and for Trials 7 vs 6 it shifts from -1 to -2. This predicts a substantial positive difference in P(A). Under the advocated SEA theory, decisions are likely to terminate early upon evaluating the highly valid first or second cue, completely ignoring the lower-validity cues that differentiate these trial pairs. Consequently, SEA predicts this difference to be near zero."
}
```

## Usage

```json
{
  "prompt_token_count": 4782,
  "candidates_token_count": 488,
  "total_token_count": 7939
}
```
