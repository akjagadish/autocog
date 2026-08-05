# metric_exp00_attempt_01

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
**Validities (n_features=6):** [0.95, 0.9, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=6):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 4: A=[1, 0, 0, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]
  trial 5: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 6: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Noisy Weighted Additive (WADD) model from Noisy Tallying, we use a 6-feature environment with a clear gap between three high-validity and three low-validity features. Tallying strictly counts the number of active features, completely ignoring validities, whereas WADD computes a weighted sum. By pitting an option with two high-validity features against one with three low-validity features, Tallying predicts a preference for the latter (3 vs. 2 wins), while WADD predicts a preference for the former (due to higher summed validities). Additional trials where Tallying predicts a tie but WADD predicts a strict preference further isolate the effect of validity weighting. Even with the high noise regimes assumed by both models, the opposing directionality of choice probabilities provides a robust dissociation.

**Computed schedule:** 6 unique pairs × 16 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Noisy Weighted Additive (WADD) Model: Decision-makers compute a global value for each option by summing the validities of its active features. Rather than relying on a single discriminating feature (like Take-The-Best) or ignoring validities (like Tallying), WADD integrates all available validities compensatorily. To account for the highly consistent ~0.5 choice probabilities and near-zero differences observed across the experiments, the decision process incorporates a highly prominent noise mechanism. This is modeled via a very low inverse temperature in the softmax choice rule and a potentially high uniform lapse rate, smoothing out predictions and avoiding deterministic swings.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.5, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # WADD: Sum of validities for active features for each option
    scores = np.sum(stim * validities, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
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
**Description:** Noisy Tallying (Equal Weights): Decision-makers ignore feature validities and instead rely on a simple tallying heuristic, counting the total number of positive features (1s) for each option. They tend to choose the option with the higher count. To account for the highly consistent ~0.5 choice probabilities observed across experiments, this process is heavily smoothed by a prominent noise mechanism, modeled via a high uniform lapse rate and a low inverse temperature in the softmax choice rule.

**Parameters:**
- beta: [0.0, 0.5]
- epsilon: [0.5, 1.0]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Tallying: Sum of active features for each option (ignoring validities)
    scores = np.sum(stim, axis=1)
    
    # Softmax with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Blend with uniform lapse rate
    n_opts = len(p_core)
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
[0] rationale: To maximally discriminate WADD from Tallying while minimizing per-subject variance, we compute the proportion of times the subject chose Option A on the three trials where WADD predicts a preference for A but Tallying predicts either a preference for B (Trial 1) or indifference (Trials 2 and 5). By pooling these three trials, we triple the number of observations per subject relative to using just Trial 1, which heavily suppresses between-subject variance and yields a much more stable metric for the Welch's t-test.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_keys = [
        (1, 1, 0, 0, 0, 0),
        (1, 0, 1, 0, 0, 0),
        (0, 0, 1, 1, 0, 0)
    ]
    data['a_key'] = data['option_a_ratings'].apply(tuple)
    mask = data['a_key'].isin(a_keys)
    relevant_data = data[mask]
    if len(relevant_data) == 0:
        return 0.5
    return float((relevant_data['response'] == 0).mean())
outcome: self_sim=0.5025 (var=0.0069) adversary_sim=0.5058 (var=0.0038) welch_t=-0.162 p=0.8724 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['a_key'] = data['option_a_ratings'].apply(tuple)\n    # Exclude Trial 6 where both models strongly prefer A\n    mask = data['a_key'] != (1, 1, 1, 0, 0, 0)\n    relevant_data = data[mask]\n    if len(relevant_data) == 0:\n        return 0.5\n    # Return the overall proportion of choosing Option A\n    return float((relevant_data['response'] == 0).mean())",
  "rationale": "To effectively discriminate WADD from Tallying despite the very high noise parameters, we calculate the aggregate proportion of choosing Option A across Trials 1 through 5. Trial 6 is excluded because both models strongly predict a preference for A. On the remaining five trials, Tallying predicts either a preference for B or indifference (tally differences of -1, 0, -1, -1, 0), resulting in an expected choice proportion for A strictly below 0.5. In contrast, WADD's validity-weighted differences on these same five trials (+0.2, +0.3, -0.25, -0.2, +0.4) sum to a positive value, yielding an expected choice proportion for A above 0.5. By pooling these five trials, we maximize the directional divergence between the two models while heavily suppressing between-subject variance."
}
```

## Usage

```json
{
  "prompt_token_count": 3384,
  "candidates_token_count": 338,
  "total_token_count": 7915
}
```
