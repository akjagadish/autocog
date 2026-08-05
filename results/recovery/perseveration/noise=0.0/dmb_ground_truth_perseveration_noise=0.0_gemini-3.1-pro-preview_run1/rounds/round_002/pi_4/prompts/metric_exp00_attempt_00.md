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
**Validities (n_features=5):** [0.95, 0.6, 0.6, 0.6, 0.6]

**Trial pairs (n=12):**
  trial 1: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 2: A=[0, 0, 0, 0, 0]  B=[1, 1, 1, 1, 1]
  trial 3: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 0, 0]
  trial 4: A=[0, 0, 0, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 5: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 6: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 9: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 0]
  trial 10: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 11: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 12: A=[0, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To conclusively demonstrate Overload/Indifference Theory over the Weighted Additive (WADD) model, we must prevent WADD from using a global 'guessing' parameter (like epsilon=1.0 or beta=0.0) to fit the data. We achieve this by mixing 'strict dominance' trials (where one option is clearly superior and subjects should rarely guess) with 'adversarial conflict' trials. In the conflict trials, a single highly predictive cue is pitted against varying numbers of weakly predictive cues (1 to 4 weak cues). WADD computes a compensatory sum based on smoothed log-odds; because the number of weak cues varies, WADD mathematically must predict a shifting preference across these conflict trials. Overload/Indifference Theory, however, predicts a flat 50/50 guessing rate across all adversarial conflicts due to cognitive overload, regardless of the exact number of weak cues. If subjects show strong preferences on dominance trials but flat 50/50 choices on all conflict trials, WADD will fail to fit the data.

**Computed schedule:** 12 unique pairs × 8 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Overload/Indifference Theory: When faced with adversarial choices where one highly valid cue conflicts with multiple weakly valid cues, subjects experience cognitive overload or perceive the options as perfectly tied. This leads to a breakdown of compensatory or non-compensatory decision strategies, resulting in pure random guessing (a 50/50 choice probability on every trial).

**Parameters:**
- dummy: {1.0}

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # Read dummy parameter to satisfy the parameter reference contract
    _ = float(parameters["dummy"])
    
    stim = np.asarray(state, dtype=float)
    n_opts = stim.shape[0]
    
    # Pure random guessing due to cognitive overload / indifference
    return np.ones(n_opts) / n_opts
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
**Description:** Weighted Additive (WADD) Model: Decision-makers evaluate options by computing a compensatory score for each, weighting every feature by its subjective importance (the log-odds of its validity). To balance extreme validities, decision-makers apply a smoothing or regularization process (Laplace smoothing) to the validities before computing their log-odds. This allows multiple weaker cues to perfectly balance a single strong cue, naturally yielding near-tied evaluations for adversarial choice pairs.

**Parameters:**
- beta: [0.0, 20.0]
- epsilon: [0.0, 1.0]
- kappa: [0.0, 10.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    kappa = float(parameters["kappa"])
    
    # Laplace smoothing to regularize extreme validities
    val_smoothed = (val + kappa) / (1.0 + 2.0 * kappa)
    
    # Transform smoothed validities to log-odds for normative compensatory weighting
    # Clip to avoid log(0) or division by zero
    val_clipped = np.clip(val_smoothed, 0.001, 0.999)
    weights = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute WADD scores as the dot product of features and weights
    scores = np.dot(stim, weights)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_sum = data['option_a_ratings'].apply(sum)\n    b_sum = data['option_b_ratings'].apply(sum)\n    dom_a = (a_sum > 0) & (b_sum == 0)\n    dom_b = (a_sum == 0) & (b_sum > 0)\n    subset = data[dom_a | dom_b]\n    if len(subset) == 0:\n        return 0.5\n    correct = ((dom_a) & (data['response'] == 0)) | ((dom_b) & (data['response'] == 1))\n    return float(correct[dom_a | dom_b].mean())",
  "rationale": "Overload/Indifference Theory predicts random guessing (0.5) across all trials, completely ignoring cue values. WADD, on the other hand, evaluates the compensatory score of each option. On pure dominance trials (where one option has positive cues and the other has none, such as trials 1-4), WADD will strongly prefer the dominant option, while Overload/Indifference Theory will still predict a 50% choice rate. This metric calculates the proportion of dominant choices on these clear trials, ensuring a large and highly significant divergence between the two models."
}
```

## Usage

```json
{
  "prompt_token_count": 3177,
  "candidates_token_count": 312,
  "total_token_count": 5859
}
```
