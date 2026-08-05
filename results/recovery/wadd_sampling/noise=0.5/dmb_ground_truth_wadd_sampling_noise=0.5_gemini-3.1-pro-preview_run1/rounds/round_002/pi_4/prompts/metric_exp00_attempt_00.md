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
**Validities (n_features=6):** [0.95, 0.85, 0.75, 0.65, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 1, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 6: A=[0, 0, 1, 1, 1, 0]  B=[1, 1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 0, 1, 1, 1]  B=[0, 1, 1, 0, 0, 0]
  trial 8: A=[0, 1, 1, 0, 0, 0]  B=[1, 0, 0, 1, 1, 1]

**Rationale:** To quantitatively dissociate the Mixture model (Advocated) from the pure WADD model (Competing), we exploit the Mixture model's independent Tallying component and its normalization of evidence. The pure WADD model uses a single non-linear transformation (gamma) to interpolate between equal weighting and lexicographic (Take-The-Best) choice. It cannot simultaneously exhibit strong sensitivity to the number of winning features (Tallying) on some trials while acting highly compensatory on others, because a single gamma parameter locks in the scaling. By contrast, the Mixture model can explicitly blend a pure count of features (Tallying) with a compensatory evaluation (WADD). We include trials where Tallying predicts a tie but WADD predicts a strong preference, trials where WADD predicts a near-tie but Tallying predicts a strong preference, and trials where the two strategies strongly oppose each other. Because the Mixture model normalizes both WADD and Tallying scores to a [0, 1] scale before applying the inverse temperature, it will uniquely capture choice probabilities that reflect an explicit blend of unweighted feature counting and normalized weighted addition.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Decision makers employ a dual-process or strategy mixture approach when evaluating multi-attribute options. Rather than relying entirely on a single strategy, choices are generated by a probabilistic mixture of a simple, unweighted Tallying heuristic (which counts the number of strictly winning features) and a compensatory Weighted Additive (WADD) strategy (which integrates all features weighted by their subjective validities). To ensure equitable application of choice determinism, the evidence scores for both strategies are normalized to a common [0, 1] scale before applying a shared inverse temperature parameter. The mixture parameter 'alpha' dictates the reliance on Tallying versus WADD, allowing the model to capture exact chance-level responding in scenarios where features tie while maintaining sensitivity to cue validities in general.

**Parameters:**
- beta: [0.01, 10.0]
- gamma: [0.0, 5.0]
- alpha: [0.0, 1.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"Expected a (2, n_features) state; got shape {stim.shape}.")
    
    a, b = stim[0], stim[1]
    n_features = stim.shape[1]
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    alpha = float(parameters["alpha"])
    epsilon = float(parameters["epsilon"])
    
    # WADD Component: Weighted sum using non-linearly scaled validities, normalized to [0, 1]
    subjective_weights = val ** gamma
    sum_weights = np.sum(subjective_weights)
    score_a_wadd = np.sum(a * subjective_weights) / sum_weights
    score_b_wadd = np.sum(b * subjective_weights) / sum_weights
    scores_wadd = np.array([score_a_wadd, score_b_wadd])
    
    z_wadd = beta * (scores_wadd - np.max(scores_wadd))
    e_wadd = np.exp(z_wadd)
    p_wadd = e_wadd / np.sum(e_wadd)
    
    # Tallying Component: Count of strict feature-wise wins, normalized to [0, 1]
    a_wins = float(np.sum(a > b)) / n_features
    b_wins = float(np.sum(b > a)) / n_features
    scores_tally = np.array([a_wins, b_wins])
    
    z_tally = beta * (scores_tally - np.max(scores_tally))
    e_tally = np.exp(z_tally)
    p_tally = e_tally / np.sum(e_tally)
    
    # Mixture of the two strategies
    p_mixed = alpha * p_tally + (1.0 - alpha) * p_wadd
    
    # Incorporate response noise (lapse rate)
    return (1.0 - epsilon) * p_mixed + epsilon * np.array([0.5, 0.5])
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
**Description:** People make choices by computing a weighted sum of all available features for each option, but the weights they use are a non-linear transformation of the objective cue validities. By exponentiating the validities with a free parameter gamma, the strategy can smoothly interpolate between equal weighting (Tallying, gamma=0) and a strong reliance on the most valid cues (approximating Take The Best, gamma > 1). The choice is then made probabilistically based on the difference between the options' weighted sums, with a bounded inverse temperature beta to allow for more stochastic choices.

**Parameters:**
- beta: [0.01, 5.0]
- epsilon: [0.0, 0.5]
- gamma: [0.0, 5.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(f"WADD expects a (2, n_features) state; got shape {stim.shape}.")
    
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities
    subjective_weights = val ** gamma
    
    # Calculate weighted sums for both options
    a, b = stim[0], stim[1]
    score_a = np.sum(a * subjective_weights)
    score_b = np.sum(b * subjective_weights)
    scores = np.array([score_a, score_b])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax choice with max-subtraction for numerical stability
    z = beta * (scores - np.max(scores))
    e = np.exp(z)
    p_core = e / np.sum(e)
    
    # Incorporate response noise (lapse rate)
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()  # Ensure valid probabilities
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data = data.copy()\n    data['A_tuple'] = data['option_a_ratings'].apply(tuple)\n    \n    # Trial 3: A=[1, 1, 1, 0, 0, 0], B=[0, 0, 0, 1, 1, 1]\n    # Trial 7: A=[1, 0, 0, 1, 1, 1], B=[0, 1, 1, 0, 0, 0]\n    t3_A = (1, 1, 1, 0, 0, 0)\n    t7_A = (1, 0, 0, 1, 1, 1)\n    \n    m3 = data[data['A_tuple'] == t3_A]['response'].mean()\n    m7 = data[data['A_tuple'] == t7_A]['response'].mean()\n    \n    # response is 0 for A, 1 for B. P(A) = 1 - mean(response)\n    p_A_t3 = 1.0 - m3\n    p_A_t7 = 1.0 - m7\n    \n    return float(p_A_t7 - p_A_t3)",
  "rationale": "This metric contrasts the choice probability of Option A between Trial 7 and Trial 3. In Trial 3, the Tallying heuristic predicts an exact tie (both options have 3 winning features), which pulls the Mixture model's prediction strongly toward 0.5. In Trial 7, Tallying strongly prefers Option A (4 wins vs 2). For the pure WADD model, the difference in weighted sums between A and B can be larger in Trial 3 than in Trial 7 (depending on the non-linear gamma parameter), meaning pure WADD can easily predict a stronger preference for A in Trial 3. By measuring P(A|Trial 7) - P(A|Trial 3), the Mixture model will consistently yield a positive value (since Tallying boosts A in Trial 7 but dilutes it in Trial 3), whereas the pure WADD model will often yield zero or negative values, providing a sharp quantitative dissociation."
}
```

## Usage

```json
{
  "prompt_token_count": 3707,
  "candidates_token_count": 533,
  "total_token_count": 7600
}
```
