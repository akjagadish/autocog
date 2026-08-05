# metric_exp00_attempt_01

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
**Validities (n_features=6):** [0.95, 0.7, 0.7, 0.7, 0.7, 0.7]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 0]
  trial 6: A=[0, 1, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0, 0]

**Rationale:** This design quantitatively dissociates the Advocated model (Weighted Additive with Power-Function Validity Scaling) from the Competing model (Take-The-Best with Compensatory Verification) by exploiting the Competing model's sub-linear tallying mechanism. The Competing model evaluates the first discriminating cue (TTB) and then applies a square-root transformation to the difference in the number of remaining winning cues. By creating a series of trials where Option A always wins the highest-validity cue, but the net difference in the remaining lower-validity cues varies from 0, to 1, to 4, the Competing model predicts equal step sizes in log-odds (sqrt(1) - sqrt(0) = 1, and sqrt(4) - sqrt(1) = 1). In contrast, the Advocated model linearly adds the exponentiated validities of these cues, predicting that the log-odds step from 1 to 4 remaining cues will be three times as large as the step from 0 to 1. This provides a strict, structural divergence between the linear evidence accumulation of WADD and the concave compensatory verification of the Competing model.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



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

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_sum = data['option_a_ratings'].apply(sum)\n    b_sum = data['option_b_ratings'].apply(sum)\n    \n    t2 = (a_sum == 2) & (b_sum == 0)\n    t3 = (a_sum == 5) & (b_sum == 0)\n    t4 = (a_sum == 1) & (b_sum == 1)\n    t5 = (a_sum == 1) & (b_sum == 4)\n    \n    def log_odds(mask):\n        subset = data[mask]\n        if len(subset) == 0:\n            return 0.0\n        count_A = (subset['response'] == 0).sum()\n        count_B = len(subset) - count_A\n        return float(np.log((count_A + 0.5) / (count_B + 0.5)))\n        \n    L2 = log_odds(t2)\n    L3 = log_odds(t3)\n    L4 = log_odds(t4)\n    L5 = log_odds(t5)\n    \n    return float(L3 - 2 * L2 + 2 * L4 - L5)",
  "rationale": "By analyzing the log-odds of choosing Option A, we can linearly separate the structural predictions of the two models. The Competing model transforms the sum of remaining cues sub-linearly (using a square root), predicting that the log-odds step from 1 to 4 remaining cues equals the step from 0 to 1. This implies L3 - L2 \u2248 L2 - L1, and symmetrically L1 - L4 \u2248 L4 - L5. By calculating L3 - 2*L2 + 2*L4 - L5, the Competing model evaluates to approximately 0. In contrast, the Advocated model assumes linear evidence accumulation, meaning the step from 1 to 4 cues is three times larger than from 0 to 1, causing this metric to deviate significantly from 0. Using smoothed empirical log-odds stabilizes the variance per subject."
}
```

## Usage

```json
{
  "prompt_token_count": 3861,
  "candidates_token_count": 525,
  "total_token_count": 7021
}
```
