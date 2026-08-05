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
**Validities (n_features=4):** [0.95, 0.85, 0.75, 0.65]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 3: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 4: A=[1, 0, 1, 0]  B=[0, 1, 0, 1]
  trial 5: A=[1, 0, 0, 1]  B=[0, 1, 1, 0]
  trial 6: A=[0, 1, 1, 0]  B=[1, 0, 0, 0]
  trial 7: A=[0, 1, 0, 1]  B=[1, 0, 0, 0]
  trial 8: A=[0, 0, 1, 1]  B=[1, 0, 0, 0]
  trial 9: A=[1, 1, 1, 0]  B=[0, 0, 0, 1]
  trial 10: A=[0, 1, 1, 1]  B=[1, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Strategy Repertoire (Mixture of TTB and WADD) from the parameterized WADD model (which uses an exponent 'gamma' to interpolate between Tallying and TTB), we use a 4-feature design with linearly spaced validities. The trials are designed to create varying degrees of conflict between the highest-validity cue and combinations of lower-validity cues. The competing WADD model applies a non-linear transformation (gamma exponent) to the weights before passing them through a single softmax function. In contrast, the Strategy Repertoire model computes a linear combination of two separate softmax probabilities (one from pure TTB, one from linear WADD). Because the softmax function is non-linear, a mixture of two distinct softmax distributions produces a probability profile across these diverse trade-off trials that cannot be perfectly mimicked by a single softmax over gamma-exponentiated weights. By analyzing the choice probabilities across trials where TTB and linear WADD strongly disagree versus where they weakly disagree, we can distinguish the mixture of strategies from a single unified compensatory strategy with non-linear weights.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Repertoire (Mixture of TTB and WADD with Independent Temperatures)

**Parameters:**
- beta_ttb: [0.1, 20.0]
- beta_wadd: [0.1, 20.0]
- p_ttb: [0.0, 1.0]
- epsilon: [0.0, 0.2]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Mixture model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    val = np.asarray(parameters["validities"], dtype=float)
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    # Take The Best (TTB) evaluation
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb == 0:
        scores_ttb = np.array([1.0, 0.0])
    elif winner_ttb == 1:
        scores_ttb = np.array([0.0, 1.0])
    else:
        scores_ttb = np.array([0.5, 0.5])
        
    # WADD evaluation (normalized by sum of validities to match TTB score scale)
    val_sum = np.sum(val)
    if val_sum > 0:
        w = val / val_sum
    else:
        w = np.ones(n_features) / n_features
        
    score_a_wadd = np.sum(w * a)
    score_b_wadd = np.sum(w * b)
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    beta_ttb = float(parameters["beta_ttb"])
    beta_wadd = float(parameters["beta_wadd"])
    p_ttb = float(parameters["p_ttb"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax for TTB
    z_ttb = beta_ttb * (scores_ttb - scores_ttb.max())
    e_ttb = np.exp(z_ttb)
    p_core_ttb = e_ttb / e_ttb.sum()
    
    # Softmax for WADD
    z_wadd = beta_wadd * (scores_wadd - scores_wadd.max())
    e_wadd = np.exp(z_wadd)
    p_core_wadd = e_wadd / e_wadd.sum()
    
    # Mixture of strategies
    p_mix = p_ttb * p_core_ttb + (1.0 - p_ttb) * p_core_wadd
    
    # Apply lapse rate
    n_opts = p_mix.shape[0]
    return (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
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
**Description:** Decision-makers integrate all available features to evaluate options, weighting each feature according to its subjective importance. This Weighted Additive (WADD) strategy computes a compensatory score for each option by summing the feature values multiplied by their normalized subjective weights. The subjective weights are modeled as the provided feature validities raised to a power 'gamma', which allows the model to smoothly interpolate between Tallying (gamma = 0, equal weights) and Take The Best (large gamma, where the highest validity cue dominates). Choice probabilities are then generated via a softmax function over the options' scores, subject to random lapses.

**Parameters:**
- gamma: [0.0, 50.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Compute subjective weights by raising validities to the power of gamma
    w = val ** gamma
    # Normalize weights to decouple gamma (cue dominance) from beta (choice determinism)
    if np.sum(w) > 0:
        w = w / np.sum(w)
    
    # Calculate Weighted Additive (WADD) scores for both options
    score_a = np.sum(w * a)
    score_b = np.sum(w * b)
    scores = np.array([score_a, score_b])
    
    # Numerically stable softmax
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    # The validities are strictly decreasing from cue 0 to cue 3.\n    # So the Take-The-Best (TTB) choice is simply determined by the first cue \n    # (from index 0 to 3) where the two options differ.\n    def get_ttb_choice(row):\n        a = row['option_a_ratings']\n        b = row['option_b_ratings']\n        for i in range(4):\n            if a[i] > b[i]: return 0\n            if b[i] > a[i]: return 1\n        return 0\n        \n    data['ttb_choice'] = data.apply(get_ttb_choice, axis=1)\n    data['is_ttb'] = (data['response'] == data['ttb_choice']).astype(float)\n    \n    # Create a hashable string representation of the trial pair to group by\n    data['trial_str'] = data.apply(lambda x: ''.join(map(str, x['option_a_ratings'])) + '_' + ''.join(map(str, x['option_b_ratings'])), axis=1)\n    \n    # Calculate the proportion of TTB-consistent choices for each unique trial type\n    trial_means = data.groupby('trial_str')['is_ttb'].mean()\n    \n    # Return the standard deviation of these proportions across the 10 trial types\n    return float(trial_means.std())",
  "rationale": "The Strategy Repertoire (Mixture) model and the non-linear WADD model produce fundamentally different distributions of choice probabilities across the 10 distinct trial types. The Mixture model combines a sharp step-function (pure TTB) with a linear compensatory function, creating a 'floor' on the TTB choice probability dictated by the mixture parameter `p_ttb`. In contrast, the non-linear WADD model applies a single exponentiation to the weights before passing them through a softmax, which can lead to more extreme variations in probabilities depending on the gamma parameter (ranging from near 0.5 for all trials when gamma is small, to almost 0 or 1 when gamma is large). By calculating the standard deviation of the TTB-consistent choice proportions across all 10 unique trial types, we quantify the dispersion of these choice probabilities, which is expected to reliably discriminate the two underlying mathematical architectures."
}
```

## Usage

```json
{
  "prompt_token_count": 3708,
  "candidates_token_count": 578,
  "total_token_count": 10214
}
```
