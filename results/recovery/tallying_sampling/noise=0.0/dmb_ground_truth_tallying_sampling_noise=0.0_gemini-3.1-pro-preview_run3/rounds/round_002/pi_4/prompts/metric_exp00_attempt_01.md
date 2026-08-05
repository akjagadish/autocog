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
**Validities (n_features=6):** [0.85, 0.8, 0.75, 0.7, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 2: A=[1, 1, 1, 1, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 3: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 1, 1, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 1, 1, 0, 0]
  trial 6: A=[1, 0, 1, 1, 1, 1]  B=[0, 1, 1, 1, 1, 1]
  trial 7: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 1, 1, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[1, 0, 0, 0, 0, 0]
  trial 10: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 1, 1, 0, 0]

**Rationale:** MCD and Tallying yield mathematically identical choice probabilities under a standard softmax choice rule for binary features, because the difference in their scores (MCD's pairwise cancellations vs. Tallying's independent sums) is always equal. However, to empirically dissociate them, we must look beyond standard choice probabilities to secondary measures (like reaction times or confidence) or explore boundary conditions where the softmax assumption breaks down. By systematically varying the number of shared positive features (shared '1's) and shared negative features (shared '0's) across trials while holding the absolute difference in scores constant, we create a testbed. If subjects use Tallying, cognitive load and perceived value might scale with the total number of '1's. If subjects use MCD, shared features are immediately canceled out, making trials with many shared '1's cognitively identical to trials with many shared '0's. This design expands on previous concepts by introducing a larger feature space (6 features) and parametric variations in both score differences and overlap.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Majority of Confirming Dimensions (MCD): People make decisions by comparing options dimension by dimension rather than computing holistic values. On each feature, the option with the superior value earns a point (a 'confirming dimension'). The option with the most points across all features is chosen. For binary features, this dimension-wise comparison yields the same relative preference as Tallying, but through a fundamentally different cognitive process (pairwise comparisons rather than independent counting).

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("MCD expects a (2, n_features) state.")
        
    a, b = stim[0], stim[1]
    
    # MCD: Count how many features strictly favor each option
    score_a = np.sum(a > b)
    score_b = np.sum(b > a)
    scores = np.array([score_a, score_b], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over MCD scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Tallying (Equal Weight Heuristic): People compare two options by ignoring the explicit validities of the cues and simply counting the number of positive features (or cues present) for each option. The option with the higher count (tally) is chosen. This is a strongly compensatory strategy that avoids the cognitive overhead of multiplying features by validities, yet allows multiple lower-validity cues to jointly overrule a single high-validity cue.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) state.")
        
    # Tallying: simple sum of positive cues per option
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over tally scores
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
[0] rationale: The experimental design aims to dissociate MCD and Tallying by varying the number of shared positive features while holding the score difference constant. Although both models yield mathematically identical choice probabilities under the provided standard softmax rule (making them indistinguishable in pure simulation), this metric formalizes the intended empirical contrast: measuring whether the choice probability of the superior option changes when cognitive load / shared features increase (Trial 3) compared to a baseline with no shared features (Trial 1).
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Since MCD and Tallying yield mathematically identical choice probabilities 
    # under the standard softmax rule for binary features, any difference in 
    # their simulated choices is purely due to sampling noise. 
    # However, to align with the experimental rationale, we measure the effect 
    # of shared positive features on the choice rate for Option A, holding the 
    # absolute score difference constant.
    
    # We identify trials based on the number of shared positive features.
    # Trial 1: A=[1,1,0,0,0,0], B=[0,0,0,0,0,0] -> Diff=2, Shared=0
    # Trial 3: A=[1,1,1,1,1,1], B=[0,0,1,1,1,1] -> Diff=2, Shared=4
    
    # Convert lists to strings for hashable comparison
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    trial_1_mask = (data['a_str'] == '110000') & (data['b_str'] == '000000')
    trial_3_mask = (data['a_str'] == '111111') & (data['b_str'] == '001111')
    
    # Mean choice of A (where response == 0)
    p_a_trial_1 = 1.0 - data[trial_1_mask]['response'].mean()
    p_a_trial_3 = 1.0 - data[trial_3_mask]['response'].mean()
    
    # If a subject has missing data for these trials, fallback to 0 difference
    if pd.isna(p_a_trial_1) or pd.isna(p_a_trial_3):
        return 0.0
        
    return float(p_a_trial_3 - p_a_trial_1)
outcome: self_sim=-0.0267 (var=0.0304) adversary_sim=-0.0311 (var=0.0277) welch_t=+0.092 p=0.9269 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    trial_1_mask = (data['a_str'] == '110000') & (data['b_str'] == '000000')\n    trial_3_mask = (data['a_str'] == '111111') & (data['b_str'] == '001111')\n    \n    p_a_trial_1 = 1.0 - data[trial_1_mask]['response'].mean()\n    p_a_trial_3 = 1.0 - data[trial_3_mask]['response'].mean()\n    \n    if pd.isna(p_a_trial_1) or pd.isna(p_a_trial_3):\n        return 0.0\n        \n    return float(p_a_trial_3 - p_a_trial_1)",
  "rationale": "As noted in the experimental rationale, MCD and Tallying yield mathematically identical choice probabilities for binary features under a standard softmax rule (because sum(A>B) - sum(B>A) is always equal to sum(A) - sum(B)). Consequently, any purely choice-based metric will produce statistically indistinguishable simulated data under these two specific model implementations. We retain the metric measuring the empirical difference in choice probability across trials with varying shared features (Trial 1 vs Trial 3) to evaluate human behavioral deviations, even though the current simulation framework cannot separate them."
}
```

## Usage

```json
{
  "prompt_token_count": 3889,
  "candidates_token_count": 416,
  "total_token_count": 10972
}
```
