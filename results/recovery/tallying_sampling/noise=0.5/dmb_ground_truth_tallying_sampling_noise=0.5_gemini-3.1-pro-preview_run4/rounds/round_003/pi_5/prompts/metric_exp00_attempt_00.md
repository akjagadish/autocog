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
**Validities (n_features=5):** [1.0, 0.9, 0.6, 0.5, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 8: A=[1, 1, 0, 1, 0]  B=[0, 0, 1, 1, 1]

**Rationale:** This design quantitatively dissociates Non-linear Validity Scaling Theory from Strategy Mixture Theory by exploiting the mathematical properties of concave scaling. Strategy Mixture Theory assumes decision-makers linearly sum cue validities (WADD) and mix this probabilistically with a Tallying (equal-weight) heuristic. In contrast, Non-linear Scaling Theory assumes validities are subjectively transformed via a fractional exponent (gamma < 1) before integration. Because a fractional exponent is a concave function, the difference between transformed low validities is larger than the difference between transformed high validities (e.g., 0.6^0.5 - 0.5^0.5 > 1.0^0.5 - 0.9^0.5). We create 'Perfect Tie' trials where Option A and Option B have identical linear WADD sums and identical Tally scores (e.g., A has validities 1.0 and 0.5; B has validities 0.9 and 0.6). Strategy Mixture predicts a perfect 50/50 tie on these trials. However, Non-linear Scaling rigidly predicts a preference for Option B, because the concave transformation boosts the 'middle' validities more than the 'extreme' validities. We also include 'Conflict' trials (where WADD favors A but Tally favors B) to ensure mixture parameters and beta can be properly identified.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Non-linear Validity Scaling Theory: Decision-makers use a single compensatory strategy to integrate cues, but they subjectively transform the provided cue validities by raising them to a fractional exponent (gamma). This non-linear scaling flattens the differences between cue validities. As gamma approaches 0, the validities become equal, naturally mimicking the Tallying (equal-weight) heuristic. As gamma approaches 1, the model recovers pure Weighted Additive (WADD) behavior. This provides a mathematically elegant, single-process account of the empirical pull towards equal weighting without assuming a discrete mixture of distinct decision strategies.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform cue validities via fractional exponent
    subjective_validities = validities ** gamma
    
    # Calculate compensatory scores
    scores = stim @ subjective_validities
    
    # Softmax choice rule with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
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
**Description:** Strategy Mixture Theory: Decision-makers are heterogeneous in their strategies. A proportion of choices are made using a compensatory Weighted Additive (WADD) strategy that integrates cue validities and subjective weights, while the remainder rely on a simpler Tallying (Equal Weight) heuristic that merely counts the number of positive features. Behavior on any given trial is a probabilistic mixture of these two strategies, with the mixture proportion varying across individuals. This naturally explains why aggregate behavior falls between the pure predictions of WADD and Tallying.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w_mix: [0.0, 1.0]
- validities: validities
- weights: [(0.0, 1.0)] * n_features

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Stimulus must be (2, n_features); got {stim.shape}")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    w = np.asarray(parameters["weights"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    w_mix = float(parameters["w_mix"])
    
    # WADD strategy component
    wadd_scores = stim @ (validities * w)
    z_wadd = beta * (wadd_scores - wadd_scores.max())
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / e_wadd.sum()
    
    # Tallying strategy component
    tally_scores = stim.sum(axis=1)
    z_tally = beta * (tally_scores - tally_scores.max())
    e_tally = np.exp(z_tally)
    p_tally = e_tally / e_tally.sum()
    
    # Mixture of strategies
    p_core = w_mix * p_wadd + (1.0 - w_mix) * p_tally
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    import pandas as pd\n    \n    validities = np.array([1.0, 0.9, 0.6, 0.5, 0.5])\n    \n    # Calculate linear WADD sums for A and B to identify perfect tie trials\n    a_sums = data['option_a_ratings'].apply(lambda x: np.dot(x, validities))\n    b_sums = data['option_b_ratings'].apply(lambda x: np.dot(x, validities))\n    \n    # Mask for trials where linear WADD and Tally both predict a tie\n    mask = np.isclose(a_sums, b_sums)\n    \n    if mask.sum() == 0:\n        return 0.5\n        \n    return float(data.loc[mask, 'response'].mean())\n",
  "rationale": "This metric isolates the 'Perfect Tie' trials (trials 1-4) where Option A and Option B have identical linear WADD scores and identical Tally scores. Under Strategy Mixture Theory, because the base strategies are perfectly indifferent, the aggregate choice proportion for Option B across subjects should be centered at 0.5 (modulo random variations in the feature-specific weights `w`). In contrast, Non-linear Validity Scaling rigidly predicts a systematic preference for Option B, because the concave fractional exponent boosts the 'middle' validities (0.9 and 0.6) more than the 'extreme' validities (1.0 and 0.5). Therefore, the mean response on these trials will be significantly greater than 0.5 for Non-linear Scaling, but approximately 0.5 for Strategy Mixture."
}
```

## Usage

```json
{
  "prompt_token_count": 3416,
  "candidates_token_count": 405,
  "total_token_count": 6631
}
```
