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
**Validities (n_features=5):** [0.95, 0.9, 0.6, 0.55, 0.5]

**Trial pairs (n=5):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 2: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]

**Rationale:** This design dissociates the Raw Validity-Weighted Additive Strategy (WADD) from the Non-linear Rank-Weighted Additive Strategy by exploiting the disconnect between raw validity differences and ordinal rank differences. We use a set of validities with a large gap between the top two cues (0.95, 0.90) and the bottom three cues (0.60, 0.55, 0.50). WADD's compensatory sums rely strictly on these raw values, making it highly sensitive to the large gap. In contrast, the Rank-Weighted model enforces a rigid, evenly-spaced ordinal structure (ranks 5, 4, 3, 2, 1) scaled by a parameter gamma. Trial pairs are constructed such that WADD's predictions (e.g., strongly preferring an option with multiple lower-ranked cues whose raw validities sum to a higher value, or predicting a tie when raw sums match) are contradicted by the Rank-Weighted model, which can amplify the highest rank (gamma > 1) to dominate the choice or compress ranks (gamma < 1) to approximate tallying.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Raw Validity-Weighted Additive Strategy: Decision-makers do not rely on fast-and-frugal heuristics or non-linear rank transformations. Instead, they compute a simple compensatory weighted sum of the features, where the weights are directly proportional to the raw validities provided in the instructions. This naturally explains the ~0.50 split in conflict trials because the sum of a few high-validity cues roughly equals the sum of several lower-validity cues, bridging the gap between heuristic and rank-based models. A constrained temperature parameter ensures that small differences in these sums translate to intermediate choice probabilities rather than deterministic outcomes.

**Parameters:**
- beta: [0.1, 5.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Compute simple weighted sums using raw validities as weights
    score_a = np.dot(a, val)
    score_b = np.dot(b, val)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    scores = np.array([score_a, score_b])
    
    # Convert scores to probabilities using softmax
    z = beta * scores
    z = z - np.max(z)
    p_core = np.exp(z)
    p_core = p_core / np.sum(p_core)
    
    n_opts = len(p_core)
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Non-linear Rank-Weighted Additive Strategy: Decision-makers evaluate options by integrating all available features, but weight them by a non-linear transformation of their ordinal rank of importance. By scaling the ranks by a power parameter gamma, the decision-maker can smoothly interpolate between pure Tallying (gamma=0, where all features are weighted equally) and steeper rank-based weighting (gamma>1). This flexibility allows the model to capture the observed ~0.50 split in conflict trials by balancing the sums of top versus bottom feature ranks appropriately.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    from scipy.stats import rankdata
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    
    # rankdata assigns rank 1 to the smallest value. 
    # Thus, higher validity gets a proportionally higher integer rank (weight).
    gamma = float(parameters["gamma"])
    weights = rankdata(val) ** gamma
    
    score_a = np.dot(a, weights)
    score_b = np.dot(b, weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    a_tuples = data['option_a_ratings'].apply(tuple)\n    b_tuples = data['option_b_ratings'].apply(tuple)\n    \n    # T5: A=[1, 0, 0, 1, 0], B=[0, 1, 0, 0, 1]\n    t5_a_mask = (a_tuples == (1, 0, 0, 1, 0)) & (b_tuples == (0, 1, 0, 0, 1))\n    t5_b_mask = (b_tuples == (1, 0, 0, 1, 0)) & (a_tuples == (0, 1, 0, 0, 1))\n    \n    t5_a_chosen = (t5_a_mask & (data['response'] == 0)).sum() + (t5_b_mask & (data['response'] == 1)).sum()\n    t5_total = t5_a_mask.sum() + t5_b_mask.sum()\n    p_t5 = t5_a_chosen / t5_total if t5_total > 0 else 0.5\n    \n    # T2: A=[0, 1, 1, 0, 0], B=[1, 0, 0, 1, 0]\n    t2_a_mask = (a_tuples == (0, 1, 1, 0, 0)) & (b_tuples == (1, 0, 0, 1, 0))\n    t2_b_mask = (b_tuples == (0, 1, 1, 0, 0)) & (a_tuples == (1, 0, 0, 1, 0))\n    \n    t2_a_chosen = (t2_a_mask & (data['response'] == 0)).sum() + (t2_b_mask & (data['response'] == 1)).sum()\n    t2_total = t2_a_mask.sum() + t2_b_mask.sum()\n    p_t2 = t2_a_chosen / t2_total if t2_total > 0 else 0.5\n    \n    return float(p_t5 - p_t2)\n",
  "rationale": "This metric contrasts the choice probabilities of the target option in Trial 5 versus Trial 2. Under the WADD (Advocated) theory, Trial 2 is a perfect tie in raw validity sums (A=1.50, B=1.50), resulting in a choice probability of exactly 0.5, while Trial 5 slightly favors A (A=1.50, B=1.40). Thus, WADD predicts a small, stable positive difference (~0.1). In contrast, the Rank-Weighted (Competing) theory applies an amplifying non-linear transformation to the ranks. For Trial 5, it heavily favors A for any gamma > 0 (as A has ranks 5 and 2 vs B's 4 and 1). For Trial 2, it actually penalizes A for gamma > 1 (as B has rank 5). Consequently, the Rank-Weighted model predicts a drastically larger difference in choice probabilities, creating a substantial and robust gap between the two theories' predictions."
}
```

## Usage

```json
{
  "prompt_token_count": 3088,
  "candidates_token_count": 816,
  "total_token_count": 17234
}
```
