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
**Validities (n_features=5):** [0.95, 0.85, 0.65, 0.55, 0.5]

**Trial pairs (n=10):**
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[0, 0, 1, 1, 1]  B=[1, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 4: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 1]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 6: A=[0, 0, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 7: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 8: A=[0, 1, 0, 1, 1]  B=[1, 0, 1, 0, 0]
  trial 9: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 10: A=[0, 0, 0, 1, 1]  B=[1, 1, 1, 0, 0]

**Rationale:** To quantitatively dissociate the Probabilistic Strategy Mixture (which occasionally uses WADD) from Pure Tallying, this design leverages 'Tally Tie' trials and 'Conflict' trials. On Tally Tie trials, both options have an equal number of positive features, but one option has higher validity features. Pure Tallying predicts exactly 50/50 guessing on these trials, whereas the Mixture model predicts a systematic preference for the higher-validity option because the WADD component breaks the tie. Furthermore, on Conflict trials, one option has more positive features (favored by Tallying) while the other has fewer but highly valid features (favored by WADD). Pure Tallying predicts a constant preference based solely on the tally difference, while the Mixture model predicts that the choice probability will be attenuated by the opposing WADD preference.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Probabilistic Strategy Mixture: Decision-makers do not blend strategies into a single value score, nor do they use deterministic tie-breaking. Instead, they probabilistically sample a decision strategy on each trial. The predominant strategy is Pure Tallying (Equal Weight), which counts positive features and ignores validities. However, on a small proportion of trials, they employ a fully compensatory Weighted Additive (WADD) strategy that integrates explicit cue validities. This mixture of pure heuristics preserves the strong tallying dominance and low validity bias in certain environments, while the occasional use of WADD produces the graded validity correlations observed in others.

**Parameters:**
- beta: [0.1, 5.0]
- p_val: [0.0, 0.5]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Expected a (2, n_features) stimulus.")
    
    # Strategy 1: Pure Tallying
    tally_scores = stim.sum(axis=1)
    
    # Strategy 2: Weighted Additive (WADD)
    validities = np.asarray(parameters["validities"], dtype=float)
    val_scores = stim @ validities
    
    beta = float(parameters["beta"])
    p_val = float(parameters["p_val"])
    epsilon = float(parameters["epsilon"])
    
    # Probabilities under Tallying
    z_t = beta * (tally_scores - np.max(tally_scores))
    e_t = np.exp(z_t)
    p_tally = e_t / e_t.sum()
    
    # Probabilities under WADD
    z_w = beta * (val_scores - np.max(val_scores))
    e_w = np.exp(z_w)
    p_wadd = e_w / e_w.sum()
    
    # Probabilistic strategy mixture
    p_mix = (1.0 - p_val) * p_tally + p_val * p_wadd
    
    # Add uniform lapse rate
    n_opts = p_mix.shape[0]
    p_final = (1.0 - epsilon) * p_mix + epsilon * (np.ones(n_opts) / n_opts)
    
    return p_final
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))
```


## COMPETING THEORY
**Description:** People make decisions between options using a Tallying (Equal Weight) heuristic. Instead of weighting features by their validities or relying on a single discriminating cue, decision-makers simply count the number of positive features for each option. They choose the option with the higher total count, treating all cues as equally important. When counts are tied, they guess. Response noise is modeled via a softmax over the tally scores and a uniform lapse rate. The choice is relatively noisy, preventing the strategy from becoming perfectly deterministic even when one option has a clear tally advantage.

**Parameters:**
- beta: [0.1, 1.5]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Tallying expects a (2, n_features) stimulus.")
    
    # Tallying: compute the sum of features for each option (equal weighting)
    scores = stim.sum(axis=1)
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Softmax over the tally scores with max-subtraction for numerical stability
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
    # Blend with uniform lapse rate
    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    validities = np.array([0.95, 0.85, 0.65, 0.55, 0.5])\n    \n    a_ratings = np.vstack(data['option_a_ratings'].values)\n    b_ratings = np.vstack(data['option_b_ratings'].values)\n    \n    tally_a = a_ratings.sum(axis=1)\n    tally_b = b_ratings.sum(axis=1)\n    \n    wadd_a = a_ratings @ validities\n    wadd_b = b_ratings @ validities\n    \n    # Focus only on trials where tally scores are tied but WADD scores differ\n    tie_mask = (tally_a == tally_b) & (wadd_a != wadd_b)\n    if not np.any(tie_mask):\n        return 0.5\n        \n    wadd_diff = wadd_a[tie_mask] - wadd_b[tie_mask]\n    wadd_favors_a = wadd_diff > 0\n    wadd_favors_b = wadd_diff < 0\n    \n    resp = data['response'].values[tie_mask]\n    \n    # Check if the subject's choice aligns with the WADD prediction\n    match_a = wadd_favors_a & (resp == 0)\n    match_b = wadd_favors_b & (resp == 1)\n    \n    return float(np.mean(match_a | match_b))",
  "rationale": "This metric isolates trials where the Pure Tallying strategy encounters a tie (both options have an equal number of positive features). On these 'Tally Tie' trials, Pure Tallying predicts random guessing, yielding an expected accuracy of 0.5 relative to any other systematic strategy. In contrast, the Probabilistic Strategy Mixture theory posits that on a subset of trials, decision-makers use a Weighted Additive (WADD) strategy that resolves these tally ties by favoring the option with higher-validity features. The metric calculates the proportion of choices that align with the WADD prediction on these tie trials. We expect Pure Tallying to score exactly ~0.5 (noise), whereas the Mixture theory will score significantly >0.5 due to the probabilistic insertion of the WADD strategy."
}
```

## Usage

```json
{
  "prompt_token_count": 3323,
  "candidates_token_count": 577,
  "total_token_count": 6628
}
```
