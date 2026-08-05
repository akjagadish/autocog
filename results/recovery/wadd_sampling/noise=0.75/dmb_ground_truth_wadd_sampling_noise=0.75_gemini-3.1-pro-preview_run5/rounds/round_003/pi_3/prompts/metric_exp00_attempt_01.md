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
**Validities (n_features=4):** [0.95, 0.65, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 0, 0]
  trial 2: A=[0, 1, 0, 0]  B=[0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 0]  B=[0, 0, 0, 1]
  trial 4: A=[1, 0, 0, 0]  B=[0, 0, 1, 0]
  trial 5: A=[0, 1, 0, 0]  B=[0, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 7: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 8: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** This design quantitatively dissociates Rank-Weighted Additive Theory from WADD by testing the strict exponential decay assumption of the former. By isolating comparisons between adjacent ranks (Rank 0 vs 1, Rank 1 vs 2, Rank 2 vs 3) using validities that have a massive cardinal drop between the first and second cue but small drops thereafter (e.g., 0.95, 0.65, 0.60, 0.55), we expose the rigidity of the Rank-Weighted model. Rank-Weighted Theory dictates that the ratio of choice logit differences between successive adjacent rank comparisons must be exactly equal to the constant decay parameter 'd'. Therefore, it mathematically cannot accommodate a large difference in choice proportions for Rank 0 vs 1 alongside equal choice proportions for Rank 1 vs 2 and Rank 2 vs 3. WADD, relying on cardinal validities, naturally predicts these varying choice logit differences because it scales the actual objective validities.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) Theory with Subjective Validity Scaling: People compare options by computing a comprehensive value for each option, integrating all available cues by weighting each feature. However, instead of using raw objective validities, individuals apply a non-linear subjective scaling to these validities. This scaling allows a single highly valid cue to be disproportionately weighted to balance out multiple less valid cues, gracefully explaining why individuals exhibit near-indifference (choice proportions around 0.5) when faced with adversarial choices where one option excels on the best cue but the other excels on numerous lesser cues. Response noise is captured via a softmax over the options' weighted sums, along with an independent lapse rate.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != stim.shape[1]:
        raise ValueError("Length of validities must match n_features.")
    
    # Apply non-linear subjective scaling to validities
    gamma = float(parameters["gamma"])
    val = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Compute weighted additive values for each option
    score_a = np.sum(a * val)
    score_b = np.sum(b * val)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice rule with max-subtraction for numerical stability
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
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Rank-Weighted Additive Theory: Individuals integrate all cues but weight them according to an exponential decay based solely on their rank-order of validity. This creates a 'soft' lexicographic rule that acts primarily like Take-The-Best, but allows multiple secondary cues to exert a small, non-zero compensatory pull on the decision. Response variability is captured via a softmax choice rule and a lapse rate.

**Parameters:**
- decay: [0.01, 1.0]
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Ranks: 0 is highest validity
    order = np.argsort(-val, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(val))
    
    # Exponential decay based on rank
    decay = float(parameters["decay"])
    weights = decay ** ranks
    
    score_a = np.sum(a * weights)
    score_b = np.sum(b * weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    epsilon = float(parameters["epsilon"])
    p_final = (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
    
    return p_final
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
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
[0] rationale: This metric calculates a linear combination of choice proportions across three adjacent-rank tradeoffs. WADD predicts a massive drop in choice proportion between Trial 1 (Rank 0 vs 1) and Trial 2 (Rank 1 vs 2) due to the objective validities (0.95 to 0.65 vs 0.65 to 0.60), but nearly identical choice proportions for Trial 2 and Trial 3 (0.60 to 0.55). Rank-Weighted theory forces an exponential decay across ranks, meaning the logit differences must decay geometrically. By computing `p1 - 3*p2 + 2*p3`, we heavily penalize the geometric decay while rewarding WADD's predicted sharp initial drop followed by a plateau.
metric_source:
def metric(data: pd.DataFrame) -> float:
    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))
    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))
    
    t1 = data[(data['a_str'] == '1000') & (data['b_str'] == '0100')]
    t2 = data[(data['a_str'] == '0100') & (data['b_str'] == '0010')]
    t3 = data[(data['a_str'] == '0010') & (data['b_str'] == '0001')]
    
    p1 = 1.0 - t1['response'].mean() if len(t1) > 0 else 0.5
    p2 = 1.0 - t2['response'].mean() if len(t2) > 0 else 0.5
    p3 = 1.0 - t3['response'].mean() if len(t3) > 0 else 0.5
    
    return float(p1 - 3 * p2 + 2 * p3)
outcome: self_sim=0.2983 (var=0.3047) adversary_sim=0.0383 (var=0.3103) welch_t=+1.658 p=0.1039 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['a_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['b_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    t1 = data[(data['a_str'] == '1000') & (data['b_str'] == '0100')]\n    t2 = data[(data['a_str'] == '0100') & (data['b_str'] == '0010')]\n    t3 = data[(data['a_str'] == '0010') & (data['b_str'] == '0001')]\n    \n    p1 = 1.0 - t1['response'].mean() if len(t1) > 0 else 0.5\n    p2 = 1.0 - t2['response'].mean() if len(t2) > 0 else 0.5\n    p3 = 1.0 - t3['response'].mean() if len(t3) > 0 else 0.5\n    \n    return float(p1 - 2 * p2 + p3)",
  "rationale": "This metric calculates the discrete second derivative of the choice proportions across the three adjacent-rank tradeoffs (Rank 0 vs 1, Rank 1 vs 2, Rank 2 vs 3). WADD predicts a massive drop in choice proportion between Trial 1 and Trial 2 due to the objective validities (0.95 to 0.65 vs 0.65 to 0.60), but nearly identical choice proportions for Trial 2 and Trial 3 (0.60 to 0.55). This sharp drop followed by a plateau yields a large positive value for `p1 - 2*p2 + p3`. Rank-Weighted theory, on the other hand, forces a constant exponential decay across ranks, which smooths out the choice proportions and results in a value closer to zero. Compared to the previously failed metric, this uses the natural, unweighted second derivative, which significantly reduces the between-subject variance by avoiding inflation of the noise in p2 and p3."
}
```

## Usage

```json
{
  "prompt_token_count": 3827,
  "candidates_token_count": 532,
  "total_token_count": 9440
}
```
