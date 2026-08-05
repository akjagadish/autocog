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
**Validities (n_features=5):** [0.95, 0.9, 0.85, 0.6, 0.55]

**Trial pairs (n=6):**
  trial 1: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate the Advocated Theory (Probabilistic Mixture of WADD and Tallying) from the Competing Theory (Lexicographic Semi-Order Evidence Accumulation with Tallying Fallback), we exploit the Competing Theory's sequential raw validity accumulation and its fallback to pure tallying. We design 'trap' trials where both the WADD component (using chance-centered validities) and the Tallying component of the Advocated Theory agree on an option, structurally forcing the Advocated Theory to predict that option. However, the Competing Theory can stop early on the first feature and predict the opposite. Conversely, we create trials where the Competing Theory is structurally forced to predict one option (because it either stops early on the first feature or falls back to a tally that favors the same option), while the Advocated Theory's WADD component strongly favors the opposite option, allowing it to predict a choice that is mathematically impossible under the Competing Theory.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Strategy Mixture (WADD and Tallying): Decision-makers consist of a population that probabilistically mixes between two distinct compensatory strategies: pure Weighted Additive (WADD) and pure Tallying. On any given decision, a subject acts as a WADD user with probability p_wadd (accumulating evidence proportional to validities centered at chance) or as a Tallying user with probability 1 - p_wadd (weighing all features equally). This directly produces the intermediate tie-breaking proportions observed in experiments without suffering from the extreme numerical distortions of exponentiated validity weights.

**Parameters:**
- p_wadd: [0.0, 1.0]
- beta: [0.1, 25.0]
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
    
    p_wadd = float(parameters["p_wadd"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # WADD component (normative weights centered at 0.5)
    w_wadd = val - 0.5
    diff_wadd = np.sum(a * w_wadd) - np.sum(b * w_wadd)
    scores_wadd = np.array([diff_wadd, -diff_wadd])
    
    z_wadd = beta * scores_wadd
    z_wadd -= np.max(z_wadd)
    e_wadd = np.exp(z_wadd)
    p_wadd_core = e_wadd / np.sum(e_wadd)
    
    # Tallying component (equal weights)
    diff_tally = np.sum(a) - np.sum(b)
    scores_tally = np.array([diff_tally, -diff_tally])
    
    z_tally = beta * scores_tally
    z_tally -= np.max(z_tally)
    e_tally = np.exp(z_tally)
    p_tally_core = e_tally / np.sum(e_tally)
    
    # Mixture of strategies
    p_core = p_wadd * p_wadd_core + (1.0 - p_wadd) * p_tally_core
    
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
**Description:** Lexicographic Semi-Order Evidence Accumulation with Tallying Fallback: Decision-makers evaluate features sequentially in order of their validity, accumulating validity-weighted evidence. If the accumulated evidence difference between the options exceeds an internal confidence threshold at any step, they stop and make a choice. If they exhaust all features without reaching the threshold (e.g., due to conflicting top features), they abandon the weighted sequential process and resort to a simple tally of all features to break the tie.

**Parameters:**
- threshold: [0.0, 5.0]
- tally_weight: [0.0, 1.5]
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
    
    # Order features by validity descending
    order = np.argsort(val)[::-1]
    
    threshold = float(parameters["threshold"])
    tally_weight = float(parameters["tally_weight"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    diff = a - b
    evidence = 0.0
    stopped = False
    
    for idx in order:
        evidence += diff[idx] * val[idx]
        if abs(evidence) >= threshold and abs(evidence) > 0:
            stopped = True
            break
            
    if not stopped:
        # Resort to a simple tally of all features if threshold not met
        evidence = np.sum(diff) * tally_weight
        
    scores = np.array([evidence, -evidence])
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_sum = data['option_a_ratings'].apply(sum)\n    b_sum = data['option_b_ratings'].apply(sum)\n    # Identify the specific trial where A=[0, 1, 1, 1, 1] and B=[1, 0, 0, 0, 0]\n    mask = (a_sum == 4) & (b_sum == 1) & (data['option_a_ratings'].apply(lambda x: x[0] == 0))\n    if not mask.any():\n        return 0.0\n    return float(data.loc[mask, 'response'].mean())",
  "rationale": "On Trial 1, Option A is [0, 1, 1, 1, 1] and Option B is [1, 0, 0, 0, 0]. The Advocated Theory's WADD component strongly prefers A (since the sum of chance-centered validities for A is positive) and its Tallying component strongly prefers A. As a result, the Advocated Theory predicts almost 0% chance of choosing B. The Competing Theory, however, evaluates the first feature (validity 0.95) which strongly favors B. If the internal confidence threshold is low enough (which happens for a subset of the population), it halts immediately and chooses B. This creates a stark divergence: the Advocated Theory predicts a response rate near 0, while the Competing Theory predicts a substantially higher rate of choosing B."
}
```

## Usage

```json
{
  "prompt_token_count": 3451,
  "candidates_token_count": 362,
  "total_token_count": 7203
}
```
