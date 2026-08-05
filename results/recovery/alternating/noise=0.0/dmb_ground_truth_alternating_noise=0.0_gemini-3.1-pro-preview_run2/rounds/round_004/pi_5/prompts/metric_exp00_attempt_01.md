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
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 1, 1]  B=[0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 3: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 5: A=[0, 0, 0, 0]  B=[1, 1, 1, 1]
  trial 6: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 7: A=[0, 0, 1, 1]  B=[1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]

**Rationale:** To dissociate Contrarian WADD from TTB with High Noise, we exploit the unique 'alpha' parameter in Contrarian WADD, which allows it to predict a preference for the option with the lower weighted sum (alpha < 0). By including dominance trials (where one option is better on all features), TTB strictly predicts the dominating option, whereas Contrarian WADD can predict the dominated option. We also include compensatory trials where the top cue conflicts with multiple lower-validity cues, and tied-sum trials where WADD predicts 50/50 but TTB strongly favors the top cue.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Contrarian WADD: Decision makers evaluate options by computing a weighted sum of their features, but they may distrust the provided expert ratings or view them as added complexity. Thus, they apply a scaling factor to the validities that can be negative, leading to an 'Anti-Tallying' or contrarian preference for options with lower scores. This weak contrarian signal is obscured by a very high rate of random guessing (lapse rate).

**Parameters:**
- alpha: [-2.0, 1.0]
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Contrarian WADD expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
        
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Compute scores with the alpha scaling factor (which can be negative)
    score_a = np.dot(stim[0], val) * alpha
    score_b = np.dot(stim[1], val) * alpha
    scores = np.array([score_a, score_b])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
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
**Description:** Take-The-Best (TTB) with High Noise: Decision makers employ a lexicographic heuristic, searching through cues in order of descending validity. They stop at the first cue that discriminates between the two options and choose the option with the higher value on that cue. However, to accommodate the empirical observation that agreement with any deterministic strategy hovers around 50%, the model incorporates a very high lapse rate (epsilon) and a low softmax inverse temperature (beta). This restricts the model to primarily exhibit random guessing, with only a weak TTB signal, matching the high degree of noise in the observed data.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.8, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    # Sort cues by descending validity; stable sort handles ties
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    a, b = stim[0], stim[1]
    winner = None
    
    # Lexicographic search
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break
            
    if winner is None:
        # No cue discriminates, guess uniformly
        return np.array([0.5, 0.5])
        
    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over binary TTB scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Apply lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
[0] rationale: This metric computes the within-subject Pearson correlation between the TTB model's binary prediction for Option A and the subject's empirical probability of choosing Option A across the 8 unique trial types.

For the 'TTB with High Noise' model, the empirical probability of choosing A will reliably be higher when TTB predicts A (due to the weak lexicographic signal), resulting in a consistently positive correlation (around +0.4 to +0.5).

For the 'Contrarian WADD' model, the alpha parameter is predominantly negative (mean -0.5), meaning the model generally prefers the option with the LOWER weighted sum of features. Because the TTB prediction is highly positively correlated with the weighted sum, Contrarian WADD's choice probabilities will generally move in the opposite direction of the TTB prediction. This results in a negative correlation on average (around -0.2 to -0.3). The substantial difference in the mean correlation (positive vs. negative) provides highly reliable discrimination between the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Convert binary rating arrays to string keys to safely identify trial types
    data['trial_type'] = data['option_a_ratings'].apply(lambda x: "".join(map(str, x))) + "_" + data['option_b_ratings'].apply(lambda x: "".join(map(str, x)))
    
    # Map each trial type to the TTB model's prediction for Option A (1 if TTB predicts A, 0 if B)
    # Lexicographic search favors the option with '1' on the first discriminating cue.
    ttb_preds = {
        "1111_0000": 1,
        "1000_0111": 1,
        "0111_1000": 0,
        "1001_0110": 1,
        "0000_1111": 0,
        "1100_0011": 1,
        "0011_1100": 0,
        "0100_0011": 1
    }
    
    df = data.copy()
    df['ttb_pred_A'] = df['trial_type'].map(ttb_preds).fillna(0.5)
    df['chose_A'] = (df['response'] == 0).astype(float)
    
    # Calculate the empirical probability of choosing A per trial type per subject
    p_A_df = df.groupby(['subject_id', 'trial_type', 'ttb_pred_A'])['chose_A'].mean().reset_index()
    
    # Calculate Pearson correlation between TTB's prediction and the subject's choice probability
    def calc_corr(subj_data):
        if len(subj_data) < 2:
            return 0.0
        c = subj_data['ttb_pred_A'].corr(subj_data['chose_A'])
        return 0.0 if pd.isna(c) else c
        
    corrs = p_A_df.groupby('subject_id').apply(calc_corr)
    
    # Return the mean correlation across all subjects
    return float(corrs.mean())
outcome: self_sim=-0.0567 (var=0.1099) adversary_sim=0.0424 (var=0.1683) welch_t=-0.939 p=0.3525 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    \n    df = data.copy()\n    \n    # Convert binary rating arrays to string keys to safely identify trial types\n    df['trial_type'] = df['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x])) + \"_\" + df['option_b_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    df['chose_A'] = (df['response'] == 0).astype(float)\n    \n    def subject_score(subj_data):\n        p_A = subj_data.groupby('trial_type')['chose_A'].mean()\n        \n        # Trial pair 1: Dominance\n        # T1: A dominates B. TTB predicts A. WADD diff A-B = +3.2\n        p_T1 = p_A.get(\"1111_0000\", 0.5)\n        # T5: B dominates A. TTB predicts B. WADD diff A-B = -3.2\n        p_T5 = p_A.get(\"0000_1111\", 0.5)\n        dX = p_T1 - p_T5\n        \n        # Trial pair 2: Conflicting cues\n        # T2: A=[1,0,0,0], B=[0,1,1,1]. TTB predicts A. WADD diff A-B = -1.3\n        p_T2 = p_A.get(\"1000_0111\", 0.5)\n        # T3: A=[0,1,1,1], B=[1,0,0,0]. TTB predicts B. WADD diff A-B = +1.3\n        p_T3 = p_A.get(\"0111_1000\", 0.5)\n        dY = p_T2 - p_T3\n        \n        # Return the product of the two differences\n        return dX * dY\n\n    scores = df.groupby('subject_id').apply(subject_score)\n    return float(scores.mean())\n",
  "rationale": "This metric leverages the structural disagreement between TTB and WADD across two carefully selected pairs of mirrored trials. \n\nLet dX be the difference in the probability of choosing A between Trial 1 (A dominates) and Trial 5 (B dominates). For TTB, dX is positive. For WADD, dX is proportional to +6.4 * alpha.\n\nLet dY be the difference in the probability of choosing A between Trial 2 (A has top cue, B has rest) and Trial 3 (B has top cue, A has rest). For TTB, the top cue dominates, so dY is positive. For WADD, the weighted sum favors the option with more cues, so dY is proportional to -2.6 * alpha.\n\nThe metric computes the product dX * dY per subject. \nFor TTB, both dX and dY are positive, so their product is consistently positive.\nFor WADD, dX and dY always have opposite signs (since one is proportional to +alpha and the other to -alpha). Thus, their product is proportional to -16.64 * alpha^2, which is strictly negative regardless of whether the subject's alpha is positive or negative. This elegantly factors out the variance of the unknown alpha parameter and cleanly separates the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 4028,
  "candidates_token_count": 848,
  "total_token_count": 16304
}
```
