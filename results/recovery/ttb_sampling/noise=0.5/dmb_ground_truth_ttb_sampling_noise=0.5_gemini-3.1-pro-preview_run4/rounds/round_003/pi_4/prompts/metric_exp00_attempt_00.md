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
**Validities (n_features=6):** [0.9, 0.84, 0.78, 0.72, 0.66, 0.6]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 1, 0, 0, 1]  B=[0, 1, 0, 1, 1, 0]
  trial 2: A=[1, 0, 0, 1, 1, 0]  B=[0, 1, 1, 0, 0, 1]
  trial 3: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 4: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 5: A=[1, 1, 0, 0, 1, 0]  B=[0, 0, 1, 1, 0, 1]
  trial 6: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]

**Rationale:** This design quantitatively dissociates Stochastic Take-The-Best (Search Error) from Strategy Selection (Mixture of TTB and WADD) by exploiting their different sensitivities to cue configurations. We use linearly spaced validities and construct matched pairs of trials (e.g., Pair 1 vs Pair 2) where the Weighted Additive (WADD) difference between Option A and Option B is mathematically identical, and standard TTB always favors Option A. Consequently, the Strategy Selection model predicts the exact same probability of choosing Option A for both trials in a pair. However, the sequence of features favoring A versus B differs in rank order between the two trials. Because Stochastic TTB's choice probabilities depend on the sequential probability of skipping features, it predicts systematically higher probabilities of choosing Option A when A's secondary favorable features appear earlier in the search order. This provides a sharp divergence in the predictions of the two models.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Stochastic Take-The-Best (Search Error)

**Parameters:**
- p_skip: [0.0, 0.5]
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
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Search through features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    p_skip = float(parameters["p_skip"])
    epsilon = float(parameters["epsilon"])
    
    p_A_core = 0.0
    p_B_core = 0.0
    prob_reach = 1.0
    
    # For each feature, there is a chance (1 - p_skip) to evaluate it correctly.
    # If it discriminates, we stop. Otherwise, or if skipped, we continue to the next.
    for idx in order:
        if a[idx] > b[idx]:
            p_A_core += prob_reach * (1.0 - p_skip)
            prob_reach *= p_skip
        elif b[idx] > a[idx]:
            p_B_core += prob_reach * (1.0 - p_skip)
            prob_reach *= p_skip
            
    # If all features are skipped or none discriminate, guess randomly
    p_A_core += prob_reach * 0.5
    p_B_core += prob_reach * 0.5
    
    p_core = np.array([p_A_core, p_B_core])
    
    # Apply general response lapse
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
**Description:** Strategy Selection (Mixture of TTB and WADD): Decision-makers are not strictly bound to a single decision strategy. Instead, there is a mixture of strategies used either across the population or within individuals on a trial-by-trial basis. Specifically, individuals choose between a non-compensatory heuristic (Take-The-Best) and a compensatory strategy (Weighted Additive Model). TTB relies solely on the highest-validity discriminating feature, while WADD computes a weighted sum of all features using their validities. The parameter 'p_wadd' dictates the probability of using WADD over TTB, allowing the model to capture both strict one-reason decision making and sensitivity to lower-validity cues when they strongly favor one option. The baseline probability of using WADD is restricted to reflect that non-compensatory heuristics are predominant.

**Parameters:**
- p_wadd: [0.1, 0.5]
- beta: [0.1, 20.0]
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
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # --- Take-The-Best (TTB) Strategy ---
    order = np.argsort(validities)[::-1]
    a_wins = 0.0
    b_wins = 0.0
    for idx in order:
        if a[idx] > b[idx]:
            a_wins = 1.0
            break
        elif b[idx] > a[idx]:
            b_wins = 1.0
            break
            
    if a_wins == 0.0 and b_wins == 0.0:
        p_ttb = np.array([0.5, 0.5])
    else:
        p_ttb = np.array([a_wins, b_wins])
        
    # --- Weighted Additive (WADD) Strategy ---
    scores = stim @ validities
    beta = float(parameters["beta"])
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_wadd_dist = e / e.sum()
    
    # --- Mixture Model ---
    p_wadd = float(parameters["p_wadd"])
    p_core = p_wadd * p_wadd_dist + (1.0 - p_wadd) * p_ttb
    
    # --- Lapse Rate ---
    epsilon = float(parameters["epsilon"])
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
(none yet)

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    a_tup = data['option_a_ratings'].apply(tuple)\n    b_tup = data['option_b_ratings'].apply(tuple)\n    \n    odd_A = {(1, 0, 1, 0, 0, 1), (1, 1, 0, 0, 0, 1), (1, 1, 0, 0, 1, 0)}\n    even_A = {(1, 0, 0, 1, 1, 0), (1, 0, 1, 0, 1, 0), (1, 0, 1, 1, 0, 0)}\n    \n    # The 'A' option in the design always has a 1 for the highest validity feature (index 0)\n    # We identify if the subject chose this 'A' option, robust to potential left/right swapping\n    chose_A = ((a_tup.apply(lambda x: x[0] == 1)) & (data['response'] == 0)) | \\\n              ((b_tup.apply(lambda x: x[0] == 1)) & (data['response'] == 1))\n              \n    is_odd = a_tup.isin(odd_A) | b_tup.isin(odd_A)\n    is_even = a_tup.isin(even_A) | b_tup.isin(even_A)\n    \n    p_a_odd = chose_A[is_odd].mean()\n    p_a_even = chose_A[is_even].mean()\n    \n    if np.isnan(p_a_odd) or np.isnan(p_a_even):\n        return 0.0\n        \n    return float(p_a_odd - p_a_even)",
  "rationale": "The experimental design constructs matched pairs of trials (e.g., Trial 1 vs Trial 2) where the Weighted Additive (WADD) difference between Option A and Option B is mathematically identical, and standard Take-The-Best (TTB) always favors Option A. The Strategy Selection model (Mixture of TTB and WADD) therefore predicts the exact same probability of choosing Option A for both trials in a pair. However, Stochastic TTB evaluates features sequentially and can skip them; it predicts a higher probability of choosing Option A when A's secondary favorable features appear earlier in the validity ranking (which happens in the 'odd' trials: 1, 3, 5). This metric calculates the difference in the probability of choosing Option A between these 'odd' trials and the matched 'even' trials (2, 4, 6). The Strategy Selection model predicts a difference of 0, whereas Stochastic TTB predicts a strictly positive difference."
}
```

## Usage

```json
{
  "prompt_token_count": 3481,
  "candidates_token_count": 679,
  "total_token_count": 8188
}
```
