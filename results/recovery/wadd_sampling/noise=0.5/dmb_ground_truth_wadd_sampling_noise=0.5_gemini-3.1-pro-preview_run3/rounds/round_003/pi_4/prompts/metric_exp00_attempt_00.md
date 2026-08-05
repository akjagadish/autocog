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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 1, 0]
  trial 2: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 8: A=[1, 0, 1, 1, 0]  B=[0, 1, 1, 1, 1]
  trial 9: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 10: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** This design aims to quantitatively dissociate Linear Weighted Additive (Linear WADD) from Tallying (Equal Weights) with a very soft softmax. Tallying simply counts the number of positive features and applies a low-beta softmax, resulting in choices that lean towards the option with more positive features but remain close to 50/50. Linear WADD, in contrast, applies a linear weight (validity - 0.5) to each feature and can have a much higher beta, leading to strong, deterministic predictions when the weighted sums differ. By using a specific validity spread [0.95, 0.85, 0.75, 0.65, 0.55], we create trials where Tallying predicts a clear advantage for one option (e.g., Option B has 3 features vs. Option A's 1 feature) while Linear WADD predicts a tie, or where Tallying predicts a tie (both have 2 features) but Linear WADD strongly predicts Option A due to higher validities. Additionally, we include reversal trials where Option B wins the tally but Option A wins the weighted linear sum.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Linear Weighted Additive Theory: Decision makers evaluate options by summing the features of each option, weighted linearly by their predictive validity (specifically, the validity's edge over chance, validity - 0.5). This represents a moderate integration strategy that avoids the extreme dominance of top cues seen in log-odds (Bayesian) weighting, while still differentiating cue importance unlike Equal-Weight/Tallying.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Linear WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Convert validities to linear weights (edge over chance)
    weights = val - 0.5
    
    a, b = stim[0], stim[1]
    
    # Calculate Weighted Additive scores for both options
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Tallying (Equal Weights) Theory with Very Soft Softmax: Decision-makers evaluate options by assigning equal weight to all features, simply counting the number of positive features for each option. However, their choices are not highly deterministic; they employ a 'very soft' decision rule where the difference in tallies yields only a mild preference (e.g., ~55/45 splits) rather than an overwhelming one, reflecting high levels of noise, uncertainty, or guessing in human decision-making in this domain.

**Parameters:**
- beta: [0.01, 0.5]
- epsilon: [0.0, 0.1]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Equal Weights expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Tallying: count the number of positive features for each option
    score_a = np.sum(a)
    score_b = np.sum(b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
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
    probabilities /= probabilities.sum()  # Ensure valid probabilities
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Extract option ratings into 2D arrays\n    A = np.stack(data['option_a_ratings'].values)\n    B = np.stack(data['option_b_ratings'].values)\n    \n    # Validities are [0.95, 0.85, 0.75, 0.65, 0.55]\n    # Linear WADD weights (validity - 0.5) are:\n    weights = np.array([0.45, 0.35, 0.25, 0.15, 0.05])\n    \n    # Calculate WADD differences\n    wadd_A = A.dot(weights)\n    wadd_B = B.dot(weights)\n    wadd_diff = wadd_A - wadd_B\n    \n    # Calculate Tallying (Equal Weights) differences\n    tally_A = A.sum(axis=1)\n    tally_B = B.sum(axis=1)\n    tally_diff = tally_A - tally_B\n    \n    # Identify \"reversal\" trials where WADD predicts A but Tallying predicts B\n    mask = (wadd_diff > 0.01) & (tally_diff < -0.01)\n    \n    if np.sum(mask) == 0:\n        return 0.5\n        \n    # Return the proportion of times subjects chose Option A (response == 0)\n    chose_A = (data['response'].values[mask] == 0)\n    return float(np.mean(chose_A))\n",
  "rationale": "This metric calculates the proportion of times a subject chooses Option A specifically on 'reversal trials'. Reversal trials are designed such that Option B has a higher total number of positive features (favored by Tallying), but Option A has a higher sum of validities (favored by Linear WADD). Because Tallying employs a soft decision rule, it will choose Option A less than 50% of the time on these trials. Conversely, Linear WADD will systematically prefer Option A, yielding a choice proportion strictly above 50% (and often much higher, depending on the beta parameter). By isolating these conflicting trials, we can cleanly discriminate between the two decision-making strategies."
}
```

## Usage

```json
{
  "prompt_token_count": 3291,
  "candidates_token_count": 571,
  "total_token_count": 6785
}
```
