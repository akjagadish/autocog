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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 1, 1, 0]  B=[1, 1, 1, 0, 0]
  trial 2: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 4: A=[1, 1, 1, 1, 1]  B=[1, 1, 1, 1, 0]
  trial 5: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 0]
  trial 6: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]

**Rationale:** To quantitatively dissociate Tallying from Cancellation and Focus, we exploit how they process shared features. Tallying computes the unweighted sum of positive features for each option and applies a softmax rule based on the difference in sums. Therefore, Tallying predicts identical choice probabilities for any trial pairs that have the same difference in total positive features, regardless of how many features the options share. Cancellation and Focus, however, first eliminates shared features and then applies a ratio rule to the remaining unique positive features. We design trials where the difference in total positive features is held constant (e.g., Option A has exactly 1 more positive feature than Option B), but the number of shared features varies. On trials where Option B's only positive features are shared with Option A, Cancellation predicts a deterministic choice for A (ratio = 1.0), whereas Tallying predicts a soft preference. On trials where the options share no features, Cancellation applies its ratio rule to a mixture of unique features, leading to a much weaker preference for A, while Tallying's prediction remains unchanged.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Tallying (Equal Weights) assumes that decision-makers rely on a simple compensatory heuristic where they count the total number of positive features for each option, completely ignoring cue validities. The option with the higher count of positive features is chosen. This equal-weighting strategy is less cognitively demanding than the Weighted Additive (WADD) rule while still allowing for compensatory decision-making, where multiple weaker cues can override a single strong cue.

**Parameters:**
- beta: [0.0, 3.0]
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    
    # Tallying: equal weights for all features, ignoring validities.
    # The score for each option is simply the sum of its positive features.
    scores = np.sum(stim, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability.
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Uniform lapse blended into the softmax
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Cancellation and Focus (Ratio Rule)

**Parameters:**
- beta: [0.0, 5.0]
- epsilon: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    # Cancellation: eliminate features shared by both options.
    # Focus only on unique positive features.
    unique_a = np.sum((stim[0] > stim[1]).astype(float))
    unique_b = np.sum((stim[1] > stim[0]).astype(float))
    
    scores = np.array([unique_a, unique_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    if unique_a == 0 and unique_b == 0:
        p_core = np.array([0.5, 0.5])
    else:
        # Ratio rule over unique features to provide a distinct alternative to Tallying's softmax.
        s_beta = np.zeros_like(scores)
        for i in range(len(scores)):
            if scores[i] > 0:
                s_beta[i] = scores[i] ** beta
            else:
                s_beta[i] = 0.0 if beta > 0 else 1.0
        p_core = s_beta / np.sum(s_beta)
        
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    a_sum = data['option_a_ratings'].apply(sum)\n    b_sum = data['option_b_ratings'].apply(sum)\n    \n    def unique_b_count(row):\n        return sum(1 for a, b in zip(row['option_a_ratings'], row['option_b_ratings']) if b == 1 and a == 0)\n        \n    unique_b = data.apply(unique_b_count, axis=1)\n    diff_sum = a_sum - b_sum\n    \n    mask1 = (diff_sum == 1) & (unique_b == 0)\n    mask2 = (diff_sum == 1) & (unique_b > 0)\n    \n    if mask1.sum() == 0 or mask2.sum() == 0:\n        return 0.0\n        \n    p_a_g1 = 1.0 - data.loc[mask1, 'response'].mean()\n    p_a_g2 = 1.0 - data.loc[mask2, 'response'].mean()\n    \n    return float(p_a_g1 - p_a_g2)",
  "rationale": "This metric calculates the difference in the probability of choosing Option A between two groups of trials. In both groups, Option A has exactly 1 more positive feature than Option B (diff_sum == 1). Tallying predicts identical choice probabilities for both groups since the difference in total positive features is the same. However, in Group 1, Option B has 0 unique features (they are all shared with A), while in Group 2, Option B has at least 1 unique feature. Cancellation and Focus predicts a deterministic choice for A in Group 1 (since B has no unique features to focus on), but a softer preference for A in Group 2 (where the ratio rule is applied to unique features). Therefore, the metric will be near 0 for Tallying and strictly positive for Cancellation."
}
```

## Usage

```json
{
  "prompt_token_count": 3065,
  "candidates_token_count": 493,
  "total_token_count": 5424
}
```
