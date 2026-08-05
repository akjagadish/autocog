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
**Validities (n_features=4):** [0.98, 0.95, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 0, 0, 0]  B=[0, 0, 0, 0]
  trial 2: A=[0, 1, 0, 0]  B=[0, 0, 0, 0]
  trial 3: A=[0, 1, 0, 0]  B=[1, 0, 0, 0]
  trial 4: A=[1, 0, 1, 1]  B=[0, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0]  B=[0, 0, 1, 1]
  trial 6: A=[0, 1, 1, 0]  B=[1, 0, 0, 1]

**Rationale:** This design quantitatively and qualitatively dissociates the advocated Weighted Additive (WADD) theory from the competing Noisy Take-The-Best (TTB) with Validity Differences theory by creating a massive inversion in cue utilization. The competing theory posits that the probability of using a cue depends strictly on the difference between its validity and the next highest validity. By setting the top two validities extremely close (e.g., 0.98 and 0.95) but creating a huge gap between the second and third (e.g., 0.95 and 0.55), the competing theory paradoxically predicts that the highest validity cue will be frequently skipped due to its small difference from the next cue, while the second cue will heavily dictate choices. WADD, conversely, scales validities absolutely via a zero-anchored exponential transformation, ensuring the highest validity cue always carries the most weight. Trials are constructed where the first and second cues conflict; WADD strictly predicts choosing the option favored by the first cue, whereas the competing theory predicts choosing the option favored by the second cue.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) Integration with Zero-Anchored Soft Validity Transformation: Decision-makers compute a subjective value for each option by summing its features, weighted by a zero-anchored exponential transformation of their validities. This transformation (exp(gamma * val) - 1) ensures that non-predictive cues receive no weight, preventing the artificial inflation of tallies by low-validity cues while allowing the highest validity cues to exponentially dominate when necessary. This naturally bridges compensatory and non-compensatory decision-making without heuristic switching.

**Parameters:**
- gamma: [0.0, 10.0]
- beta: [0.1, 50.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Subjective transformation of validities
    # Subtracting 1.0 ensures that a zero-validity cue would receive exactly 0 weight,
    # preventing artificial inflation of low-validity cues and allowing the highest
    # validity cues to dominate when necessary.
    w = np.exp(gamma * val) - 1.0
    
    # Compute subjective values (Weighted Additive sum)
    v_a = np.sum(w * a)
    v_b = np.sum(w * b)
    
    scores = np.array([v_a, v_b])
    
    # Softmax choice rule
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
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
**Description:** Noisy Take-The-Best (Probabilistic Lexicographic Choice) with Validity Differences

**Parameters:**
- alpha: [0.0, 10.0]
- gamma: [0.1, 5.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    alpha = float(parameters["alpha"])
    gamma = float(parameters["gamma"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity descending
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    p_a = 0.0
    p_b = 0.0
    p_skip_accum = 1.0
    
    for i, j in enumerate(cue_order):
        if a[j] != b[j]:
            if i < len(cue_order) - 1:
                next_j = cue_order[i+1]
                diff = val[j] - val[next_j]
            else:
                diff = val[j]
                
            # Probability of successfully using this cue scales with its validity difference
            p_success = alpha * (diff ** gamma)
            p_success = min(max(p_success, 0.0), 1.0)
            
            if a[j] > b[j]:
                p_a += p_skip_accum * p_success
            else:
                p_b += p_skip_accum * p_success
                
            p_skip_accum *= (1.0 - p_success)
            
    # If all discriminating cues are skipped or no cues discriminate, guess randomly
    p_a += p_skip_accum * 0.5
    p_b += p_skip_accum * 0.5
    
    # Apply lapse rate
    p_a = (1.0 - epsilon) * p_a + epsilon * 0.5
    p_b = (1.0 - epsilon) * p_b + epsilon * 0.5
    
    return np.array([p_a, p_b])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    \n    # Identify trials where cue 1 and cue 2 conflict\n    conflict = (a_ratings[:, 0] != b_ratings[:, 0]) & \\\n               (a_ratings[:, 1] != b_ratings[:, 1]) & \\\n               (a_ratings[:, 0] != a_ratings[:, 1])\n               \n    if not np.any(conflict):\n        return 0.5\n        \n    responses = data['response'].values\n    \n    a_has_cue1 = a_ratings[:, 0] > b_ratings[:, 0]\n    b_has_cue1 = b_ratings[:, 0] > a_ratings[:, 0]\n    \n    # Check if the chosen option is the one favored by cue 1\n    chose_cue1 = (a_has_cue1 & (responses == 0)) | (b_has_cue1 & (responses == 1))\n    \n    return float(np.mean(chose_cue1[conflict]))",
  "rationale": "The experimental design creates a conflict between the first and second most valid cues. Under the advocated WADD theory, the exponential transformation ensures that the highest validity cue consistently outweighs the second, leading to choices that align with cue 1. In contrast, the competing Noisy Take-The-Best theory evaluates cue usage probabilistically based on validity differences. Because the difference between the first and second validities (0.98 - 0.95 = 0.03) is minuscule compared to the difference between the second and third (0.95 - 0.55 = 0.40), the competing theory predicts that subjects will frequently skip the first cue and rely on the second cue instead. Thus, measuring the proportion of choices aligned with cue 1 on conflict trials will yield a value near 1 for WADD and near 0 for Noisy TTB."
}
```

## Usage

```json
{
  "prompt_token_count": 3239,
  "candidates_token_count": 502,
  "total_token_count": 5879
}
```
