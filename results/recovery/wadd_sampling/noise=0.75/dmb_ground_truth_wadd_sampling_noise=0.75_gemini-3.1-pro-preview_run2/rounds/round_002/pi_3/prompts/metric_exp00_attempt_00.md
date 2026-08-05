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
**Validities (n_features=5):** [0.85, 0.84, 0.83, 0.55, 0.51]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 6: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Weighted Additive (WADD) theory from the Strategy Mixture Theory (TTB + Tallying), we use a 5-feature design with three high-validity cues with very similar validities and two low-validity cues. Because WADD uses the log-odds of validities as weights, the combined weight of the second and third most valid cues strictly outweighs the single most valid cue combined with all low-validity cues. In critical trials, Option A possesses the highest validity cue alongside multiple low-validity cues, while Option B possesses only the second and third most valid cues. TTB will select Option A because it has the best cue. Tallying will also select Option A because it has more cues in total (e.g., 3 vs. 2). Consequently, any mixture of TTB and Tallying must strictly prefer Option A. In stark contrast, WADD will strongly prefer Option B, providing a definitive dissociation. Additional trials pit TTB and Tallying against each other in ways that align WADD with one or the other, ensuring robust parameter estimation across theories.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People make decisions by computing a subjective value for each option through a Weighted Additive (WADD) process. They take a weighted sum of the features for each option, where the weight of each feature is the log-odds of its provided cue validity. This transformation appropriately scales probabilities into additive evidence. The probability of choosing an option is then determined by a softmax over the options' subjective values, combined with a lapse rate for random errors. This compensatory mechanism allows multiple lower-validity cues to jointly outweigh a single high-validity cue, naturally predicting the graded, intermediate behavior observed on conflict trials compared to non-compensatory heuristics like Take The Best or Tallying.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    # Transform validities to log-odds to represent additive evidence
    # Clip to avoid division by zero or log(0)
    val_clipped = np.clip(val, 1e-5, 1.0 - 1e-5)
    w = np.log(val_clipped / (1.0 - val_clipped))
    
    # Compute weighted additive values for both options
    v_a = np.sum(a * w)
    v_b = np.sum(b * w)
    scores = np.array([v_a, v_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax with max-subtraction for numerical stability
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
    probabilities = probabilities / np.sum(probabilities)
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Strategy Mixture Theory: People do not rely exclusively on a single decision strategy. Instead, individuals or trials draw from a repertoire of strategies, specifically a mixture of a non-compensatory heuristic (Take The Best) and a compensatory heuristic (Tallying). The probability of using TTB versus Tallying is governed by a mixture parameter, allowing the model to capture intermediate choice proportions on conflict trials where the two strategies make opposite predictions. Tallying treats all cues equally, ensuring it provides a genuine compensatory contrast to TTB.

**Parameters:**
- beta: [0.01, 10.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be a (2, n_features) array.")
    
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    p_ttb_weight = float(parameters["p_ttb"])
    
    # --- Take The Best (TTB) Prediction ---
    cue_order = np.argsort(-val, kind="stable").tolist()
    winner_ttb = None
    for j in cue_order:
        if a[j] > b[j]:
            winner_ttb = 0
            break
        if b[j] > a[j]:
            winner_ttb = 1
            break
            
    if winner_ttb is None:
        p_ttb = np.array([0.5, 0.5])
    else:
        scores_ttb = np.array([1.0, 0.0]) if winner_ttb == 0 else np.array([0.0, 1.0])
        z_ttb = beta * (scores_ttb - np.max(scores_ttb))
        e_ttb = np.exp(z_ttb)
        p_ttb = e_ttb / np.sum(e_ttb)
        
    # --- Tallying (Compensatory) Prediction ---
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores_comp = np.array([a_wins, b_wins])
    
    z_comp = beta * (scores_comp - np.max(scores_comp))
    e_comp = np.exp(z_comp)
    p_comp = e_comp / np.sum(e_comp)
    
    # --- Mixture ---
    p_mixed = p_ttb_weight * p_ttb + (1.0 - p_ttb_weight) * p_comp
    
    n_opts = p_mixed.shape[0]
    return (1.0 - epsilon) * p_mixed + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / np.sum(probabilities)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    target_profile = (0, 1, 1, 0, 0)\n    matches = 0\n    total = 0\n    for _, row in data.iterrows():\n        a = tuple(row['option_a_ratings'])\n        b = tuple(row['option_b_ratings'])\n        if a == target_profile:\n            total += 1\n            if row['response'] == 0:\n                matches += 1\n        elif b == target_profile:\n            total += 1\n            if row['response'] == 1:\n                matches += 1\n    if total == 0:\n        return 0.5\n    return float(matches / total)",
  "rationale": "This metric calculates the proportion of times the subject selects the option with the feature profile [0, 1, 1, 0, 0] when it is available (which occurs in Trials 1, 2, and 3). Under the WADD theory, the combined weight of cues 2 and 3 (the second and third most valid cues) strictly outweighs the combined weight of cue 1 and the low-validity cues, leading to a strong preference for the [0, 1, 1, 0, 0] option. In contrast, the Strategy Mixture Theory (TTB + Tallying) will strongly prefer the alternative option, because TTB favors the option with the best cue (cue 1), and Tallying either favors the alternative (which has more cues in total) or is indifferent. Thus, WADD predicts a value close to 1, while the Mixture theory predicts a value close to 0."
}
```

## Usage

```json
{
  "prompt_token_count": 3558,
  "candidates_token_count": 394,
  "total_token_count": 6586
}
```
