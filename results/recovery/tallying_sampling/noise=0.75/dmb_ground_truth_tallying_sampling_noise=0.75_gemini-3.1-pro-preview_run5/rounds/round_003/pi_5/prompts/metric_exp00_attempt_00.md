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
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[0, 0, 0, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 4: A=[1, 0, 0, 1, 0]  B=[0, 1, 0, 0, 1]
  trial 5: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 8: A=[1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1]

**Rationale:** To quantitatively dissociate pure Tallying from the advocated Zero-Centered Tallying Interpolation (WADD) model, we use a 5-feature design with a linear spread of validities. Pure Tallying evaluates options strictly based on the difference in the raw count of positive features, completely ignoring validities. The advocated model predicts choices based on a blend of uniform weights and zero-centered validities. The trials systemically pit the tally difference against the zero-centered validity difference. We include 'tally-tied' trials where zero-centered validities strongly favor one option (Tallying predicts guessing, WADD predicts a deterministic choice), and 'tally-advantaged' trials where the option with more positive cues is strongly opposed by high-validity cues (Tallying predicts the option with more cues, WADD predicts the option with fewer but more valid cues).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive Strategy with Zero-Centered Tallying Interpolation. Decision-makers integrate all available cues using a compensatory strategy, but the subjective weights applied to the cues are a blend of uniform weighting (pure Tallying) and zero-centered objective validities. This allows behavior to smoothly transition from simply counting features (when cues are treated equally) to a fully validity-sensitive linear WADD model with high disparity between cues, capturing both the strong reliance on total feature counts in some contexts and the extreme validity-driven asymmetries in others.

**Parameters:**
- alpha: [0.0, 1.0]
- beta: [0.0, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    val = np.asarray(parameters["validities"], dtype=float)
    alpha = float(parameters["alpha"])
    
    # Blend between uniform weights (Tallying) and zero-centered validities (WADD)
    weights = (1.0 - alpha) * 1.0 + alpha * (val - 0.5)
    
    # Compensatory integration: sum of weighted cues
    scores = np.sum(stim * weights, axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the integrated scores
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate for choice noise
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
```


## COMPETING THEORY
**Description:** Tallying (Equal-Weight) Heuristic: People evaluate options by simply counting the number of positive features (or cues favoring each option) and choosing the option with the higher total count. This compensatory strategy ignores the differential validities or subjective importance of different cues, treating all pieces of evidence equally. The choice probability is determined by a softmax over the total feature tallies for each option, combined with a uniform lapse rate. Crucially, the softmax temperature is constrained to produce softer choice probabilities, reflecting that humans do not apply the tallying rule completely deterministically.

**Parameters:**
- beta: [0.0, 1.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tally the number of positive cues for each option
    score_a = np.sum(stim[0])
    score_b = np.sum(stim[1])
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tallies
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probs = np.asarray(probabilities, dtype=np.float64)
    probs /= probs.sum()
    return np.random.choice(len(probs), p=probs)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    validities = np.array([0.95, 0.85, 0.75, 0.65, 0.55])\n    \n    # Extract option ratings as 2D arrays\n    A = np.stack(data['option_a_ratings'].values)\n    B = np.stack(data['option_b_ratings'].values)\n    \n    # Calculate tallies for each option\n    sum_A = np.sum(A, axis=1)\n    sum_B = np.sum(B, axis=1)\n    \n    # Isolate \"tally-tied\" trials where Tallying predicts guessing\n    tied_mask = (sum_A == sum_B)\n    \n    if not np.any(tied_mask):\n        return 0.5\n        \n    # Calculate WADD values for the tied trials\n    val_A = np.sum(A[tied_mask] * validities, axis=1)\n    val_B = np.sum(B[tied_mask] * validities, axis=1)\n    \n    responses = data['response'].values[tied_mask]\n    \n    # Determine the option favored by the WADD model (0 for A, 1 for B)\n    expected = (val_B > val_A).astype(int)\n    \n    # Return the proportion of choices aligning with the WADD model's prediction\n    return float(np.mean(responses == expected))",
  "rationale": "This metric isolates 'tally-tied' trials (where both options have the same number of positive cues). On these trials, the pure Tallying model predicts that subjects will guess randomly (resulting in an expected alignment proportion of 0.5). In contrast, the advocated Zero-Centered Tallying Interpolation (WADD) model predicts that subjects will consistently favor the option with the higher objective validity sum. By calculating the proportion of choices that align with the WADD prediction specifically on tally-tied trials, we can maximally discriminate the two theories."
}
```

## Usage

```json
{
  "prompt_token_count": 3131,
  "candidates_token_count": 502,
  "total_token_count": 6414
}
```
