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
**Validities (n_features=5):** [0.95, 0.93, 0.91, 0.6, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 2: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Additive Utility with chance-centered validity weighting) from the Competing Theory (Rank-Based Weighting), we exploit the difference between magnitude-based and ordinal-based feature evaluation. The Advocated Theory scales weights based on the absolute magnitude of the validities, meaning that features with very similar validities will receive almost identical weights. In contrast, the Competing Theory relies strictly on ordinal ranks, enforcing a fixed drop-off in weight between rank 1, 2, and 3, regardless of how close their actual validities are. By using a set of validities with a tight cluster at the top (e.g., 0.95, 0.93, 0.91) and a drop-off to lower validities (0.60, 0.55), we construct trials where the sum of the 2nd and 3rd best features easily outweighs the 1st best feature in the Advocated Theory (because 0.93 and 0.91 are almost as good as 0.95), but fails to do so in the Competing Theory (because rank 2 and 3 are heavily penalized compared to rank 1). Conversely, we create trials where multiple low-rank features can outweigh a single higher-rank feature in the Competing Theory, but fail to do so in the Advocated Theory due to the massive gap in objective validity.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** People use a single-stage Additive Utility evaluation where features are weighted by a power-law transformation of their chance-centered objective validities. By avoiding artificial weight normalization and allowing for sufficiently high softmax temperatures and power parameters, the decision-making process can smoothly and stably interpolate between Tallying (equal weights), proportional weighting, and highly deterministic Take-The-Best behavior (where the most valid cues dominate completely).

**Parameters:**
- beta: [0.1, 50.0]
- gamma: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Transform validities using a simple power law on chance-centered values
    # gamma=0 -> Tallying, gamma=1 -> WADD, gamma>1 -> TTB
    centered_val = val - 0.5
    weights = (centered_val + 1e-6) ** gamma
    
    scores = np.array([np.sum(a * weights), np.sum(b * weights)])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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


## COMPETING THEORY
**Description:** Rank-Based Weighting with Normalized Power-Law Decay: Decision makers rank features by their objective validities and assign decision weights based on their ordinal rank. By applying a power-law transformation on the ranks (e.g., w_i = rank_i ^ -gamma) and explicitly normalizing these weights to sum to 1, the model ensures that the total accumulated evidence remains bounded on a consistent scale regardless of the steepness parameter gamma. This allows a single temperature parameter to stably govern choice determinism across both strictly compensatory (Tallying) and non-compensatory (Take-The-Best) strategies.

**Parameters:**
- beta: [0.1, 100.0]
- gamma: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Rank validities descending (highest validity gets rank 1)
    ranks = rankdata(-val)
    
    # Ordinal weights: w_i = rank_i ^ (-gamma)
    # Features with validity exactly 0.5 are chance and should be ignored
    weights = np.where(val > 0.5, ranks ** (-gamma), 0.0)
    
    # Normalize weights so they sum to 1.0, keeping scores on a consistent scale
    sum_w = np.sum(weights)
    if sum_w > 0:
        weights = weights / sum_w
    
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_str = data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    b_str = data['option_b_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x]))\n    \n    is_trial_1 = ((a_str == \"01100\") & (b_str == \"10000\")) | ((a_str == \"10000\") & (b_str == \"01100\"))\n    is_trial_3 = ((a_str == \"10011\") & (b_str == \"01100\")) | ((a_str == \"01100\") & (b_str == \"10011\"))\n    \n    t1_data = data[is_trial_1]\n    t3_data = data[is_trial_3]\n    \n    if len(t1_data) == 0 or len(t3_data) == 0:\n        return 0.0\n        \n    a_is_23_t1 = t1_data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x])) == \"01100\"\n    chose_23_t1 = (a_is_23_t1 & (t1_data['response'] == 0)) | (~a_is_23_t1 & (t1_data['response'] == 1))\n    \n    a_is_23_t3 = t3_data['option_a_ratings'].apply(lambda x: \"\".join([str(int(v)) for v in x])) == \"01100\"\n    chose_23_t3 = (a_is_23_t3 & (t3_data['response'] == 0)) | (~a_is_23_t3 & (t3_data['response'] == 1))\n    \n    return float(chose_23_t1.mean() + chose_23_t3.mean())",
  "rationale": "The metric computes the overall proportion of times subjects choose the option with features 2 and 3 (the '01100' option) in Trials 1 and 3. Under the Advocated Theory, validities 0.93 and 0.91 are almost as strong as 0.95, so their combined weight easily outweighs feature 1 (0.95) in Trial 1, and also outweighs features 1, 4, 5 (0.95, 0.60, 0.55) in Trial 3. Thus, the Advocated Theory consistently prefers the '01100' option in both trials, leading to a metric value near 2.0. In contrast, the Competing Theory enforces a strict ordinal drop-off, heavily penalizing ranks 2 and 3. As a result, feature 1 dominates features 2 and 3 in Trial 1, and features 1, 4, 5 dominate features 2 and 3 in Trial 3. The Competing Theory therefore avoids the '01100' option in both trials, yielding a metric value near 0.0."
}
```

## Usage

```json
{
  "prompt_token_count": 3342,
  "candidates_token_count": 809,
  "total_token_count": 7846
}
```
