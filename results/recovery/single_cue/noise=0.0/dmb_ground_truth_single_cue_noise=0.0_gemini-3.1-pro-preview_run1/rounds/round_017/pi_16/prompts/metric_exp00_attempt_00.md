# metric_exp00_attempt_00

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
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.6, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 1, 0, 0, 1]  B=[1, 0, 1, 1, 0]
  trial 2: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 0, 0, 1, 1]  B=[1, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Context-Dependent Mixture of TTB and Tallying with Inverse Validity Tie-Breaking) from the Competing Theory (Soft Primacy-Biased Sequential Integration), we exploit their fundamentally different treatments of explicit validities versus spatial position. The Competing Theory uses an exponential left-to-right primacy decay for its WADD component and completely ignores explicit validities. The Advocated Theory uses explicit validities for both its mixture weight and its inverse-validity tie-breaker (which strongly favors lower-validity features). By setting validities to monotonically decrease from left to right, the Competing Theory's primacy bias aligns with the highest validities, while the Advocated Theory's inverse-validity tie-breaker strongly opposes it by favoring the right-most (lowest validity) features. We include 'Tally Tie' trials where TTB and Primacy WADD both favor the left-winning option, but the Inverse Validity tie-breaker favors the right-winning option, creating a sharp divergence. We also include 'Compensatory' trials where the left-most feature competes against multiple right-most features to further highlight the difference between Primacy WADD and pure Tallying.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Context-Dependent Dual-Process Mixture of TTB and Tallying with Inverse Validity Tie-Breaking: Decision-makers rely on a mixture of Take-The-Best (TTB) and Tallying, but the mixture weight is dynamically determined by the environment. When cue validities are highly dispersed (measured by the standard deviation of the validities), subjects predominantly use TTB; when validities are similar, they rely on Tallying. When Tallying results in a tie, subjects resolve it using an inverse-validity weighting mechanism, heavily favoring options with positive features among the lower-validity (or more recently processed) cues.

**Parameters:**
- validities: validities
- disp_slope: [0.0, 100.0]
- disp_threshold: [0.0, 1.0]
- w_tie: [0.0, 0.95]
- gamma: [0.1, 10.0]
- beta_tally: [0.1, 20.0]
- beta_ttb: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    validities = np.asarray(parameters["validities"], dtype=float)
    disp_slope = float(parameters["disp_slope"])
    disp_threshold = float(parameters["disp_threshold"])
    w_tie = float(parameters["w_tie"])
    gamma = float(parameters["gamma"])
    beta_tally = float(parameters["beta_tally"])
    beta_ttb = float(parameters["beta_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Calculate mixture weight based on dispersion of validities (standard deviation)
    dispersion = float(np.std(validities))
    w_ttb = 1.0 / (1.0 + np.exp(-disp_slope * (dispersion - disp_threshold)))
    
    # --- Strategy 1: Tallying with Inverse Validity Tie-Breaker ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    tie_weights = (1.0 - validities) ** gamma
    if np.sum(tie_weights) > 0:
        tie_weights /= np.sum(tie_weights)
    else:
        tie_weights = np.ones_like(tie_weights) / len(tie_weights)
        
    tie_score_a = np.sum(a * tie_weights)
    tie_score_b = np.sum(b * tie_weights)
    
    # w_tie < 1.0 ensures the tie-breaker only dictates choice when a_wins == b_wins
    score_a_tally = a_wins + w_tie * tie_score_a
    score_b_tally = b_wins + w_tie * tie_score_b
    
    scores_tally = np.array([score_a_tally, score_b_tally])
    z_tally = beta_tally * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    probs_tally = e_tally / np.sum(e_tally)
    
    # --- Strategy 2: Take-The-Best (TTB) ---
    ttb_a = 0.0
    ttb_b = 0.0
    for i in range(n_features):
        if a[i] > b[i]:
            ttb_a = 1.0
            break
        elif b[i] > a[i]:
            ttb_b = 1.0
            break
            
    scores_ttb = np.array([ttb_a, ttb_b])
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_ttb = np.exp(z_ttb)
    probs_ttb = e_ttb / np.sum(e_ttb)
    
    # --- Mixture Model ---
    mixed_probs = w_ttb * probs_ttb + (1.0 - w_ttb) * probs_tally
    
    # --- Lapse Rate ---
    final_probs = (1.0 - epsilon) * mixed_probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** Soft Primacy-Biased Sequential Integration

**Parameters:**
- w_ttb: [0.0, 0.1]
- gamma: [0.0, 0.5]
- beta_ttb: [0.1, 20.0]
- beta_wadd: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    n_features = len(a)
    
    w_ttb = float(parameters["w_ttb"])
    gamma = float(parameters["gamma"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    epsilon = float(parameters["epsilon"])
    
    # Strategy 1: Take-The-Best (TTB)
    # Process from left to right and terminate at the first discriminating feature
    ttb_a = 0.0
    ttb_b = 0.0
    for i in range(n_features):
        if a[i] > b[i]:
            ttb_a = 1.0
            break
        elif b[i] > a[i]:
            ttb_b = 1.0
            break
            
    scores_ttb = np.array([ttb_a, ttb_b])
    z_ttb = beta_ttb * (scores_ttb - np.max(scores_ttb))
    e_ttb = np.exp(z_ttb)
    probs_ttb = e_ttb / np.sum(e_ttb)
    
    # Strategy 2: Primacy-Boosted WADD
    # Compensatory mechanism with exponentially decaying weights from left to right
    ranks = np.arange(n_features)
    weights = np.exp(-gamma * ranks)
    if np.sum(weights) > 0:
        weights /= np.sum(weights)
    else:
        weights = np.ones_like(weights) / n_features
    
    score_a_wadd = np.sum(a * weights)
    score_b_wadd = np.sum(b * weights)
    
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    z_wadd = beta_wadd * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    probs_wadd = e_wadd / np.sum(e_wadd)
    
    # Mixture Model
    mixed_probs = w_ttb * probs_ttb + (1.0 - w_ttb) * probs_wadd
    
    # Lapse Rate
    final_probs = (1.0 - epsilon) * mixed_probs + epsilon * np.array([0.5, 0.5])
    
    return final_probs
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify Trial 2: A=[0, 1, 0, 1, 0] vs B=[0, 0, 1, 0, 1]\n    t2_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 1, 0, 1, 0))\n    # Identify Trial 5: A=[0, 0, 0, 1, 1] vs B=[1, 1, 0, 0, 0]\n    t5_mask = data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 0, 1, 1))\n    \n    p_b_t2 = data.loc[t2_mask, 'response'].mean()\n    p_b_t5 = data.loc[t5_mask, 'response'].mean()\n    \n    # Fallback for empty slices\n    if pd.isna(p_b_t2):\n        p_b_t2 = 0.5\n    if pd.isna(p_b_t5):\n        p_b_t5 = 0.5\n        \n    return float(p_b_t2 - p_b_t5)",
  "rationale": "The metric computes the difference in the probability of choosing option B between Trial 2 and Trial 5. In Trial 2, the Competing Theory's TTB and Primacy WADD both strongly favor option A, leading to a near-zero probability of choosing B. In Trial 5, both mechanisms strongly favor option B, leading to a near-one probability of choosing B. Therefore, the Competing Theory predicts a metric value close to -1.0. In contrast, the Advocated Theory's inverse-validity tie-breaker pulls the predictions in the exact opposite directions (favoring B in Trial 2 and A in Trial 5), which shrinks the difference considerably toward 0 or even makes it positive, providing a stark and stable contrast between the two models."
}
```

## Usage

```json
{
  "prompt_token_count": 4078,
  "candidates_token_count": 490,
  "total_token_count": 7565
}
```
