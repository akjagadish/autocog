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
  trial 1: A=[1, 0, 0, 0, 1]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[1, 0, 1, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 4: A=[0, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[0, 0, 1, 0, 0]  B=[0, 1, 0, 0, 1]
  trial 6: A=[1, 0, 0, 0, 1]  B=[0, 0, 1, 0, 0]
  trial 7: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 1]  B=[1, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the pure Averaging model (Advocated) from the Lexicographic with Fallback model (Competing), we exploit their differing reliance on the single best discriminating cue. The Competing model uses a threshold ('theta') on the highest-validity discriminating cue to probabilistically trigger a Take-The-Best (TTB) choice; if triggered, it ignores all other features. The pure Averaging model, however, always evaluates options based on the mean validity of their positive features. We design trials where Option A possesses the highest-validity discriminating cue (strongly triggering TTB in the Competing model) but is paired with low-validity features that heavily 'dilute' its overall average. Option B lacks the top cue but possesses one or more moderately high-validity features, resulting in a higher average validity. The Competing model will frequently choose the option with the top cue due to its TTB component, while the pure Averaging model will consistently prefer the option with the higher mean validity.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Averaging Model: Decision-makers evaluate options not by summing the evidence (additive) or relying on a single best cue (lexicographic), but by averaging the subjective validities of all positive features present in an option. This mechanism naturally accounts for the 'dilution effect', where adding a positive but low-validity feature to an option can actually decrease its overall subjective value and choice probability.

**Parameters:**
- gamma: [1.0, 10.0]
- beta: [0.1, 15.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Transform validities to subjective weights
    w = val ** gamma
    
    # Count positive features
    k_a = np.sum(a)
    k_b = np.sum(b)
    
    # Calculate average validity of positive features (0 if no positive features)
    score_a = np.sum(w * a) / k_a if k_a > 0 else 0.0
    score_b = np.sum(w * b) / k_b if k_b > 0 else 0.0
    
    scores = np.array([score_a, score_b])
    
    # Softmax choice rule
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


## COMPETING THEORY
**Description:** Lexicographic with Fallback (Probabilistic TTB + Averaging): Decision-makers process features sequentially in order of validity. If the most valid discriminating feature provides strong enough evidence (its validity exceeds a threshold), they rely on it to make a Take-The-Best choice. However, this TTB choice is probabilistic, with its reliability scaled by the cue's validity, allowing for variance across different cue validities. If the best discriminating feature is not decisive enough, or as a probabilistic alternative, they fall back to a holistic evaluation by averaging the subjective validities of all positive features.

**Parameters:**
- gamma: [0.0, 10.0]
- beta_avg: [0.1, 20.0]
- beta_ttb: [0.1, 20.0]
- p_ttb: [0.0, 1.0]
- theta: [0.5, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Model expects a (2, n_features) stimulus.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta_avg = float(parameters["beta_avg"])
    beta_ttb = float(parameters["beta_ttb"])
    p_ttb = float(parameters["p_ttb"])
    theta = float(parameters["theta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending (stable sort to preserve original order on ties)
    cue_order = np.argsort(-val, kind="stable")
    
    # Find first discriminating cue
    j_star = -1
    for j in cue_order:
        if a[j] != b[j]:
            j_star = j
            break
            
    # Lexicographic (TTB) Choice formulation
    if j_star != -1:
        v_star = val[j_star]
        # Probabilistic TTB using softmax scaled by the discriminating cue's validity
        z_ttb = beta_ttb * v_star * np.array([a[j_star], b[j_star]])
        z_ttb = z_ttb - np.max(z_ttb)
        e_ttb = np.exp(z_ttb)
        p_core_ttb = e_ttb / np.sum(e_ttb)
    else:
        p_core_ttb = np.array([0.5, 0.5])
        v_star = 0.0
        
    # Decide whether to use TTB based on the decisiveness (threshold) of the discriminating cue
    if v_star >= theta:
        w_ttb = p_ttb
    else:
        w_ttb = 0.0
        
    # Fallback Strategy: Averaging of all positive features
    w = val ** gamma
    k_a = np.sum(a)
    k_b = np.sum(b)
    
    score_a = np.sum(w * a) / k_a if k_a > 0 else 0.0
    score_b = np.sum(w * b) / k_b if k_b > 0 else 0.0
    
    scores = np.array([score_a, score_b])
    
    # Softmax for Averaging fallback
    z = beta_avg * (scores - np.max(scores))
    e = np.exp(z)
    p_core_avg = e / np.sum(e)
    
    # Mixture of Lexicographic and Fallback strategies
    p_mix = w_ttb * p_core_ttb + (1.0 - w_ttb) * p_core_avg
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    val = np.array([0.95, 0.85, 0.75, 0.65, 0.55])\n    \n    A = np.stack(data['option_a_ratings'].values)\n    B = np.stack(data['option_b_ratings'].values)\n    \n    # Determine TTB choice (first discriminating cue)\n    diff = A - B\n    idx = np.argmax(np.abs(diff), axis=1)\n    ttb_choice = np.where(diff[np.arange(len(diff)), idx] == 1, 0, 1)\n    \n    # Determine Averaging choice\n    sum_a = np.sum(A, axis=1)\n    sum_b = np.sum(B, axis=1)\n    \n    avg_a = np.zeros(len(A))\n    mask_a = sum_a > 0\n    avg_a[mask_a] = np.sum(A[mask_a] * val, axis=1) / sum_a[mask_a]\n    \n    avg_b = np.zeros(len(B))\n    mask_b = sum_b > 0\n    avg_b[mask_b] = np.sum(B[mask_b] * val, axis=1) / sum_b[mask_b]\n    \n    avg_choice = np.where(avg_a > avg_b, 0, np.where(avg_b > avg_a, 1, -1))\n    \n    # Filter trials where TTB and Averaging disagree\n    disagree_mask = (avg_choice != -1) & (ttb_choice != avg_choice)\n    \n    if not np.any(disagree_mask):\n        return 0.5\n        \n    responses = data['response'].values[disagree_mask]\n    ttb_choices = ttb_choice[disagree_mask]\n    \n    return float(np.mean(responses == ttb_choices))\n",
  "rationale": "This metric isolates trials where the Take-The-Best (TTB) heuristic and the Averaging heuristic strictly disagree on which option is superior. It calculates the proportion of times the subject's response aligns with the TTB prediction on these conflicting trials. The Averaging model should produce values close to 0 (or random if noise is high), whereas the Lexicographic with Fallback model should produce substantially higher values due to its probabilistic reliance on the single best discriminating cue."
}
```

## Usage

```json
{
  "prompt_token_count": 3834,
  "candidates_token_count": 626,
  "total_token_count": 7870
}
```
