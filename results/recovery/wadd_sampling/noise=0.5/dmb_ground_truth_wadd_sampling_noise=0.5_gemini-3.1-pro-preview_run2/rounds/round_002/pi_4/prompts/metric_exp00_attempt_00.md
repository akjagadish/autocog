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
**Validities (n_features=5):** [0.9, 0.8, 0.7, 0.6, 0.5]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 0]
  trial 2: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 5: A=[0, 1, 0, 1, 0]  B=[1, 0, 0, 0, 1]
  trial 6: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 7: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 8: A=[1, 0, 0, 1, 1]  B=[0, 1, 1, 0, 0]

**Rationale:** To cleanly dissociate the Advocated Hybrid Heuristic Theory from the Competing Weighted Additive (WADD) Theory with Non-Linear Cue Scaling, we exploit the functional form of their effective cue weights. The Hybrid theory assumes a linear combination of normalized WADD (which is linear in validity) and Tallying (which applies a uniform weight to all features). Consequently, the Hybrid model's effective weights are always strictly linear with respect to cue validities. In contrast, the Competing WADD theory scales validities by an exponent (gamma), producing weights that are convex (gamma > 1) or concave (gamma < 1) with respect to validity. We use 5 features with linearly spaced validities to test this. For example, in trials where the sum of validities is equal (e.g., cues 1 and 4 vs. cues 2 and 3), the Hybrid model perfectly ties regardless of its mixing parameter, whereas the Non-Linear WADD model will predictably diverge from a tie depending on its gamma parameter. By including a mix of these 'linear-tie' trials alongside classical Tallying vs. linear WADD contradictions, we can perfectly isolate the linear interpolation of the Hybrid model from the non-linear scaling of the Competing model.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Hybrid Heuristic Theory: Decision-makers integrate evidence from multiple strategies before making a choice, rather than probabilistically sampling between distinct strategies. Specifically, individuals compute a combined subjective value for each option by taking a weighted average of normalized compensatory (validity-weighted) and non-compensatory (tallying) evidence. A single stochastic decision process then operates on these integrated values.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- w_mix: [0.0, 1.0]
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
    n_features = len(val)
    
    # Normalized WADD scores
    sum_val = np.sum(val)
    score_wadd_a = np.sum(val * a) / sum_val
    score_wadd_b = np.sum(val * b) / sum_val
    
    # Normalized Tallying scores
    score_tal_a = float(np.sum(a > b)) / n_features
    score_tal_b = float(np.sum(b > a)) / n_features
    
    beta = float(parameters["beta"])
    w_mix = float(parameters["w_mix"])
    epsilon = float(parameters["epsilon"])
    
    # Combined integrated scores
    score_combined_a = w_mix * score_wadd_a + (1.0 - w_mix) * score_tal_a
    score_combined_b = w_mix * score_wadd_b + (1.0 - w_mix) * score_tal_b
    scores = np.array([score_combined_a, score_combined_b])
    
    # Single Softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p = e / np.sum(e)
    
    # Lapse rate
    return (1.0 - epsilon) * p + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** Weighted Additive (WADD) Theory with Non-Linear Cue Scaling: People evaluate multi-attribute options by computing an overall value for each option. This value is determined by taking a weighted sum of the option's features, where the weights correspond to the subjective validities of the respective cues scaled by a non-linear parameter. This scaling allows decision-makers to flexibly upweight highly valid cues (approaching a non-compensatory strategy) or downweight them (approaching an equal-weight tallying strategy). Decision-makers then choose probabilistically between the options by comparing these total weighted scores via a softmax function.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities
- gamma: [0.1, 6.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) stimulus; got shape {stim.shape}.")
        
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities
    weights = val ** gamma
    
    a, b = stim[0], stim[1]
    
    # Calculate weighted sums for both options
    score_a = np.sum(weights * a)
    score_b = np.sum(weights * b)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate lapse rate
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
```

**`policy source code`:**
```python
def policy(probabilities):
    import numpy as np
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(np.random.choice(len(probabilities), p=probabilities))
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    import numpy as np\n    \n    # Convert rating lists to string representations to identify specific trials\n    data['A_str'] = data['option_a_ratings'].apply(lambda x: ''.join(map(str, x)))\n    data['B_str'] = data['option_b_ratings'].apply(lambda x: ''.join(map(str, x)))\n    \n    # Identify the 'linear-tie' trials\n    # Trial 1: Cues 1 & 4 vs Cues 2 & 3 (sum of validities = 1.5 for both)\n    mask_t1 = (data['A_str'] == '10010') & (data['B_str'] == '01100')\n    # Trial 5: Cues 2 & 4 vs Cues 1 & 5 (sum of validities = 1.4 for both)\n    mask_t5 = (data['A_str'] == '01010') & (data['B_str'] == '10001')\n    \n    dev = 0.0\n    if mask_t1.sum() > 0:\n        p_A_t1 = (data.loc[mask_t1, 'response'] == 0).mean()\n        dev += abs(p_A_t1 - 0.5)\n        \n    if mask_t5.sum() > 0:\n        p_A_t5 = (data.loc[mask_t5, 'response'] == 0).mean()\n        dev += abs(p_A_t5 - 0.5)\n        \n    return float(dev)",
  "rationale": "In the advocated Hybrid theory, the effective weights of the cues are strictly linear with respect to their validities. Thus, for trials where the sum of validities of the present cues is equal for both options (Trials 1 and 5), the Hybrid model predicts an exact tie (P(Choose A) = 0.5) regardless of the mixing parameter or softmax temperature. In contrast, the competing WADD theory applies a non-linear scaling (gamma) to the validities. Because x^gamma is strictly convex or concave for gamma != 1, the scaled sums will diverge from a tie, pushing the choice probability away from 0.5. This metric computes the absolute deviation of the choice probability from 0.5 on these specific linear-tie trials. It will be near 0 (modulo sampling noise) for the Hybrid model, but significantly larger for the WADD model."
}
```

## Usage

```json
{
  "prompt_token_count": 3431,
  "candidates_token_count": 613,
  "total_token_count": 7998
}
```
