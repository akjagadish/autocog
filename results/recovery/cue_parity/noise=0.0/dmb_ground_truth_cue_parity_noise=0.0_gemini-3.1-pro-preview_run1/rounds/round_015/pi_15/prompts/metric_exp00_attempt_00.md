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
**Validities (n_features=5):** [0.95, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]

**Rationale:** This design quantitatively dissociates the Advocated Theory (Top-K Majority Heuristic) from the Competing Theory (Pure Tallying with Validity-Weighted Tie-Breaking) by exploiting the difference between evaluating a fixed subset of top features versus tallying all features. We construct trials where one option wins the overall unweighted tally across all 5 features (so the Competing Theory predicts it will win), but the other option holds a majority within the top 3 most valid features (so the Advocated Theory predicts it will win). Furthermore, we include 'tie' trials where both options win on an equal number of features overall. In these cases, the Competing Theory falls back on a validity-weighted sum across all features, while the Advocated Theory strictly tallies within the Top-K subset, leading to diametrically opposed predictions.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Top-K Majority Heuristic with Proportional Confidence: Decision-makers evaluate options by considering a subset of the most valid features. They identify the top K most valid features and perform an unweighted tally within this set. If tied, they may expand the set. Their confidence in the choice scales with the proportional majority margin (vote difference divided by K), meaning a given vote margin yields higher confidence in smaller consideration sets.

**Parameters:**
- k_initial: [1.0, 10.0]
- expand_on_tie: [0.0, 1.0]
- beta: [0.1, 15.0]
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
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    # Determine initial consideration set size K
    k_init = int(round(float(parameters["k_initial"]))) # e.g., 3 or 5
    k = min(k_init, len(val))
    k = max(1, k)
    
    expand_on_tie = float(parameters["expand_on_tie"]) > 0.5
    
    while True:
        top_k_idx = order[:k]
        a_wins = np.sum(a[top_k_idx] > b[top_k_idx])
        b_wins = np.sum(b[top_k_idx] > a[top_k_idx])
        
        # Stop if there's a strict majority winner within top K
        if a_wins != b_wins:
            break
            
        # If tied, either expand K or accept the tie (and guess)
        if expand_on_tie and k < len(val):
            k += 1
        else:
            break
            
    diff = float(a_wins - b_wins) / k
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the vote difference
    scores = np.array([diff, -diff])
    z = beta * scores
    z -= np.max(z)  # numerical stability
    e = np.exp(z)
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
**Description:** Pure Tallying with Validity-Weighted Tie-Breaking: Decision-makers primarily rely on an unweighted tally of positive features to evaluate options. They simply count the number of positive features for each option and select the one with the higher count, effectively ignoring feature validities in the first pass. This explains the strong preference for the majority option even when it lacks the most valid feature. However, if the tallies are tied, they fall back to a compensatory strategy, computing a validity-weighted sum of the features to break the tie.

**Parameters:**
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- tie_scale: [0.1, 20.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) stimulus; got shape {stim.shape}.")
        
    a, b = stim[0], stim[1]
    
    # Primary stage: Unweighted tally of positive features
    a_tally = float(np.sum(a))
    b_tally = float(np.sum(b))
    
    if a_tally != b_tally:
        scores = np.array([a_tally, b_tally])
    else:
        # Secondary stage: Validity-weighted tie-breaker
        val = np.asarray(parameters["validities"], dtype=float)
        tie_scale = float(parameters["tie_scale"])
        scores = tie_scale * np.array([np.sum(a * val), np.sum(b * val)])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * scores
    z -= np.max(z)
    e = np.exp(z)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify Trial 1 where A=[0, 0, 1, 1, 1] and B=[1, 1, 0, 0, 0]\n    t1_mask = (data['option_a_ratings'].apply(lambda x: tuple(x) == (0, 0, 1, 1, 1))) & \\\n              (data['option_b_ratings'].apply(lambda x: tuple(x) == (1, 1, 0, 0, 0)))\n    if not t1_mask.any():\n        return 0.0\n    # Return proportion of times option A was chosen (response == 0)\n    return float((data.loc[t1_mask, 'response'] == 0).mean())",
  "rationale": "On Trial 1, Option A wins the overall tally across all 5 features (3 vs 2), so the Competing Theory (Pure Tallying) overwhelmingly predicts A. However, Option B wins within the top 2 and top 3 most valid features. The Advocated Theory (Top-K Majority), depending on the subject's consideration set size K, will frequently predict B. Therefore, the proportion of choosing A on this trial will be near 1.0 for the Competing Theory, but significantly lower for the Advocated Theory."
}
```

## Usage

```json
{
  "prompt_token_count": 3345,
  "candidates_token_count": 323,
  "total_token_count": 6262
}
```
