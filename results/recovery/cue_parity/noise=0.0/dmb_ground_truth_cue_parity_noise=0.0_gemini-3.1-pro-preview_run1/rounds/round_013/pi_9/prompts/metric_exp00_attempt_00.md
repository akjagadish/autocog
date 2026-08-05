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
**Validities (n_features=6):** [0.95, 0.94, 0.93, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 2: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1, 1, 1]  B=[1, 1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 1, 0, 1, 0]  B=[0, 1, 0, 1, 0, 1]
  trial 6: A=[0, 1, 0, 1, 0, 1]  B=[1, 0, 1, 0, 1, 0]

**Rationale:** To quantitatively dissociate the Advocated Theory (Unweighted Sequential Evidence Accumulation) from the Competing Theory (Weighted Sequential Evidence Accumulation with Self-Excitation/Decay), we exploit the differences between unweighted discrete accumulation (+1/-1) and weighted, leaky continuous accumulation. By using a tight cluster of high validities (0.95, 0.94, 0.93) and a drop-off to lower validities (0.60, 0.55, 0.50), we construct trials where the Advocated Theory's unweighted running tally fluctuates but structurally forces it to predict one option (or guess) regardless of its threshold. For instance, if the difference sequence is [1, -1, -1, 1, 1, 1], the unweighted tally never drops below -1 and finishes at +2, meaning the Advocated Theory must choose Option A. However, the Competing Theory weights features by a power of their validities, allowing the 2nd and 3rd features to overwhelmingly overpower the 1st, 4th, 5th, and 6th, predicting a strong preference for Option B. Furthermore, we include trials like [1, 1, 1, -1, -1, -1] where the Advocated Theory either stops early (choosing A) or tallies to a tie (guessing), but the Competing Theory's 'leak' parameter can cause early evidence to decay, allowing the later features to dominate and resulting in a robust choice for Option B. This creates a sharp double dissociation across multiple parameter regimes.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



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
**Description:** Weighted Sequential Evidence Accumulation with Self-Excitation/Decay (Leaky Accumulator). Decision-makers process information sequentially in descending order of feature validity. Each feature's evidence is weighted by its validity and added to a running accumulator. Crucially, previously accumulated evidence can either 'leak' (decay) or self-excite (amplify) as new features are evaluated. A choice is made either when the accumulator hits a predefined threshold or when all features are exhausted. Self-excitation allows the model to capture strong primacy effects even when the decision threshold is not reached.

**Parameters:**
- threshold: [0.0, 5.0]
- leak: [-1.0, 1.0]
- gamma: [0.0, 5.0]
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
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["threshold"])
    leak = float(parameters["leak"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Evaluate features in descending order of validity
    order = np.argsort(val)[::-1]
    
    A = 0.0
    for idx in order:
        diff = a[idx] - b[idx]
        weight = val[idx] ** gamma
        
        # Leaky accumulation (or self-excitation if leak < 0)
        A = A * (1.0 - leak) + weight * diff
        
        # Stopping rule
        if abs(A) >= threshold and abs(A) > 1e-9:
            break
            
    scores = np.array([A, -A])
    
    # Softmax conversion to probabilities
    z = beta * scores
    z -= np.max(z)  # For numerical stability
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    # Extract features to isolate Trials 1 and 2\n    a_f1 = data['option_a_ratings'].apply(lambda x: x[0])\n    b_f1 = data['option_b_ratings'].apply(lambda x: x[0])\n    a_f2 = data['option_a_ratings'].apply(lambda x: x[1])\n    a_f4 = data['option_a_ratings'].apply(lambda x: x[3])\n    \n    # Trial 1: a=[1, 0, 0, 1, 1, 1], b=[0, 1, 1, 0, 0, 0]\n    # Trial 2: a=[0, 1, 1, 0, 0, 0], b=[1, 0, 0, 1, 1, 1]\n    # In both cases, feature 1 matches feature 4 for option A, and differs from feature 2.\n    mask = (a_f1 == a_f4) & (a_f1 != a_f2) & (a_f1 != b_f1)\n    subset = data[mask]\n    \n    if len(subset) == 0:\n        return 0.5\n        \n    # Calculate if the choice aligns with the first (most valid) feature\n    a_is_better_on_f1 = subset['option_a_ratings'].apply(lambda x: x[0]) > subset['option_b_ratings'].apply(lambda x: x[0])\n    b_is_better_on_f1 = subset['option_a_ratings'].apply(lambda x: x[0]) < subset['option_b_ratings'].apply(lambda x: x[0])\n    \n    chose_a = subset['response'] == 0\n    chose_b = subset['response'] == 1\n    \n    aligned = (a_is_better_on_f1 & chose_a) | (b_is_better_on_f1 & chose_b)\n    \n    return float(aligned.mean())",
  "rationale": "This metric calculates the proportion of choices that align with the most valid feature (feature 1) specifically on Trials 1 and 2. In these trials, the unweighted accumulator (Advocated Theory) either stops early on feature 1 or tallies all the way to the end where the +1s outnumber the -1s, consistently choosing the option favored by feature 1. In contrast, the weighted leaky accumulator (Competing Theory) allows the closely matched validities of features 2 and 3 to overpower feature 1, particularly when early evidence decays (leak), leading to a high rate of choosing the opposite option."
}
```

## Usage

```json
{
  "prompt_token_count": 3491,
  "candidates_token_count": 679,
  "total_token_count": 7950
}
```
