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

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 3: A=[0, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]
  trial 6: A=[1, 0, 1, 0, 1]  B=[0, 1, 0, 1, 0]
  trial 7: A=[1, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Sequential Evidence Accumulation) from the Competing Theory (Two-Phase Heuristic: TTB with Tallying Fallback), we exploit their divergent behavior depending on whether the most valid feature is tied. The Competing Theory strictly uses the best feature if it discriminates; if tied, it abandons sequential search and tallies the rest. The Advocated Theory strictly accumulates differences (+1/-1) sequentially and stops if a threshold is reached. By constructing trials where the first feature is tied but the second feature strongly favors Option A while the remaining features favor Option B, the Competing Theory tallies and chooses B, whereas the Advocated Theory (with a low threshold) stops early on the second feature and chooses A. Conversely, when the first feature is NOT tied but all other features favor the opposite option, the Competing Theory chooses the TTB option, while the Advocated Theory (with a high threshold) will tally and choose the opposite.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision-makers use a Sequential Evidence Accumulation strategy with a stopping rule. They inspect features one by one in descending order of their validity, maintaining a running sum of the differences between the options. If the absolute accumulated evidence reaches or exceeds a specific threshold, they stop and make a choice based on that evidence. If the threshold is not reached, they evaluate all features and decide based on the final tally. This allows the model to smoothly transition between Take-The-Best (low threshold) and Tallying (high threshold) behaviors.

**Parameters:**
- threshold: [0.0, 3.0]
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
    
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = a - b
    accumulated_evidence = 0.0
    
    for idx in order:
        accumulated_evidence += diff[idx]
        if abs(accumulated_evidence) >= threshold and abs(accumulated_evidence) > 0:
            break
            
    scores = np.array([accumulated_evidence, -accumulated_evidence])
    
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
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
**Description:** Two-Phase Heuristic (Take-The-Best with Tallying Fallback): Decision-makers first evaluate the single most valid feature. If it discriminates between options, they make a choice immediately based on it, with the determinism of this choice modulated by a specific weight. If the most valid feature is tied, they abandon sequential lexicographic evaluation and instead perform an unweighted Tally of all remaining features to break the tie.

**Parameters:**
- beta: [0.1, 15.0]
- ttb_weight: [0.1, 10.0]
- tally_scale: [0.1, 10.0]
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
    
    # Identify the most valid feature
    best_idx = np.argmax(val)
    diff_best = a[best_idx] - b[best_idx]
    
    if diff_best != 0:
        # Phase 1: The most valid feature discriminates
        ttb_weight = float(parameters["ttb_weight"])
        scores = ttb_weight * np.array([diff_best, -diff_best])
    else:
        # Phase 2: The most valid feature is tied; Tally the remaining features
        mask = np.ones(len(a), dtype=bool)
        mask[best_idx] = False
        tally_a = np.sum(a[mask])
        tally_b = np.sum(b[mask])
        
        tally_scale = float(parameters["tally_scale"])
        scores = tally_scale * np.array([tally_a, tally_b])
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
    z = beta * scores
    z -= np.max(z)
    p_core = np.exp(z) / np.sum(np.exp(z))
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n\n    df = data.copy()\n    # Convert response to +1 for choosing A, -1 for choosing B\n    df['C'] = 1 - 2 * df['response']\n    \n    # Hashable representations\n    df['a_tup'] = df['option_a_ratings'].apply(tuple)\n    df['b_tup'] = df['option_b_ratings'].apply(tuple)\n    \n    def get_c(a_target, b_target):\n        mask1 = (df['a_tup'] == a_target) & (df['b_tup'] == b_target)\n        mask2 = (df['a_tup'] == b_target) & (df['b_tup'] == a_target)\n        \n        c_sum = 0.0\n        if mask1.sum() > 0:\n            c_sum += df.loc[mask1, 'C'].sum()\n        if mask2.sum() > 0:\n            c_sum -= df.loc[mask2, 'C'].sum()\n            \n        total = mask1.sum() + mask2.sum()\n        return c_sum / total if total > 0 else 0.0\n\n    # Trial 1: F0 discriminates (A is better), rest favor B\n    c1 = get_c((1,0,0,0,0), (0,1,1,1,1))\n    # Trial 2: F0 tied, F1 favors A, rest favor B\n    c2 = get_c((1,1,0,0,0), (1,0,1,1,1))\n    # Trial 3: F0 tied, F1 favors B, rest favor A\n    c3 = get_c((0,0,1,1,1), (0,1,0,0,0))\n    # Trial 4: F0 discriminates (B is better), rest favor A\n    c4 = get_c((0,1,1,1,0), (1,0,0,0,1))\n    \n    return float(c1 * c2 + c3 * c4)",
  "rationale": "This metric exploits a structural dissociation in the choice patterns between the Advocated Theory (Sequential Evidence Accumulation) and the Competing Theory (Two-Phase Heuristic). By converting choices to +1 (Option A) and -1 (Option B), we examine the product of average choice probabilities for specific trial pairs. For the Competing Theory, the choice flips depending on whether the most valid feature is tied or discriminates, resulting in negative products (C1*C2 < 0 and C3*C4 < 0). In contrast, the Advocated Theory aligns its choices based on its stopping threshold: for a low threshold it stops early on both trials, and for a high threshold it tallies on both, producing identical choice directions in both cases. Thus, the Advocated Theory yields positive products (C1*C2 > 0 and C3*C4 > 0). Summing these products yields a positive value for the Advocated Theory and a negative value for the Competing Theory, perfectly discriminating them across all parameter regimes."
}
```

## Usage

```json
{
  "prompt_token_count": 3399,
  "candidates_token_count": 770,
  "total_token_count": 10251
}
```
