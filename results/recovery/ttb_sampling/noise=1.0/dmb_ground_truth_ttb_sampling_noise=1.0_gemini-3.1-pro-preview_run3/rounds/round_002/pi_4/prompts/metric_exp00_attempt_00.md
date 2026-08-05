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
**Validities (n_features=5):** [0.95, 0.75, 0.73, 0.71, 0.69]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 0, 1, 0, 0]  B=[0, 1, 0, 1, 1]
  trial 4: A=[0, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 5: A=[1, 0, 0, 1, 0]  B=[0, 1, 1, 0, 1]
  trial 6: A=[0, 1, 0, 1, 0]  B=[0, 0, 1, 0, 1]
  trial 7: A=[1, 0, 0, 0, 1]  B=[0, 1, 1, 1, 0]
  trial 8: A=[0, 1, 0, 0, 1]  B=[0, 0, 1, 1, 0]

**Rationale:** This design leverages a structural invariance in the Strategy Mixture model to cleanly dissociate it from Noisy TTB. We use a 5-feature environment where the most valid feature is highly diagnostic (0.95), while the remaining four features are clustered tightly together (0.75, 0.73, 0.71, 0.69). In Trial 1, Option A is supported only by the top feature (0.95), while Option B is supported by all four lower features. In Trial 2, Option A is supported only by the second-best feature (0.75), while Option B is supported by the bottom three. According to the Strategy Mixture model, both trials present the exact same conflict: strict TTB chooses Option A, and Tallying chooses Option B. Thus, the Strategy Mixture model predicts identical choice probabilities for Option A across both trials. In contrast, Noisy TTB predicts a much higher choice probability for Option A in Trial 1 (because the 0.95 validity is far enough above the rest to resist noise-induced rank swapping) than in Trial 2 (where the 0.75 validity is easily swapped with the tightly clustered lower features, leading to frequent choices of Option B). Additional trials vary the tallying margins to further constrain parameters.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Strategy Mixture: Decision-makers are heterogeneous and use a mixture of non-compensatory and compensatory heuristics. On any given trial, a decision is made either via Take-The-Best (relying solely on the most valid discriminating feature) or Tallying (counting the total number of winning features), governed by a mixing parameter. When these strategies conflict, their opposing choices average out across the population, naturally capturing the ~0.5 aggregate choice proportions observed across experiments without relying on massive uniform noise.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- p_ttb: [0.0, 1.0]
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
    p_ttb = float(parameters["p_ttb"])
    
    # Take-The-Best (TTB) predictions
    order = np.argsort(validities)[::-1]
    ttb_scores = np.zeros(2)
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            ttb_scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            ttb_scores[1] = 1.0
            break
            
    z_ttb = beta * (ttb_scores - np.max(ttb_scores))
    e_ttb = np.exp(z_ttb)
    p_ttb_dist = e_ttb / e_ttb.sum()
    
    # Tallying predictions
    a_wins = float(np.sum(stim[0] > stim[1]))
    b_wins = float(np.sum(stim[1] > stim[0]))
    tally_scores = np.array([a_wins, b_wins])
    
    z_tally = beta * (tally_scores - np.max(tally_scores))
    e_tally = np.exp(z_tally)
    p_tally_dist = e_tally / e_tally.sum()
    
    # Mixture of the two strategies
    p_core = p_ttb * p_ttb_dist + (1.0 - p_ttb) * p_tally_dist
    
    # Uniform lapse
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
**Description:** Take-The-Best (TTB) heuristic with subjective validity noise: Decision-makers search through features in descending order of their subjective validity, but their assessment of validity is noisy. On any given evaluation, they sample validities with some variance, sort features by these sampled validities, and stop at the first feature that discriminates between the two options. Response noise is modeled via a softmax over the resulting binary preference scores, plus an independent lapse rate. Averaging over multiple samples naturally smooths the aggregate predictions when validities are close.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- sigma: [0.0, 2.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
    
    validities = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    sigma = float(parameters["sigma"])
    
    n_samples = 50
    avg_p_core = np.zeros(2)
    
    for _ in range(n_samples):
        noisy_v = validities + np.random.normal(0, sigma, size=validities.shape)
        order = np.argsort(noisy_v)[::-1]
        
        scores = np.zeros(2)
        # Search for the first discriminating feature
        for idx in order:
            if stim[0, idx] > stim[1, idx]:
                scores[0] = 1.0
                break
            elif stim[1, idx] > stim[0, idx]:
                scores[1] = 1.0
                break
                
        # Softmax over the scores
        z = beta * (scores - scores.max())
        e = np.exp(z)
        p_core = e / e.sum()
        avg_p_core += p_core
        
    avg_p_core /= n_samples
    
    n_opts = avg_p_core.shape[0]
    return (1.0 - epsilon) * avg_p_core + epsilon * (np.ones(n_opts) / n_opts)
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    # Convert ratings to a 2D numpy array\n    a_ratings = np.stack(data['option_a_ratings'].values)\n    \n    # Trials 1, 3, 5, 7 have the top feature (index 0) active for Option A\n    # Trials 2, 4, 6, 8 have the second-best feature (index 1) active for Option A, and index 0 is tied\n    is_a0_1 = a_ratings[:, 0] == 1\n    \n    chose_a = (data['response'] == 0).values\n    \n    # Calculate the proportion of times Option A was chosen in each condition\n    p_a_when_top_feature = np.mean(chose_a[is_a0_1])\n    p_a_when_second_feature = np.mean(chose_a[~is_a0_1])\n    \n    return float(p_a_when_top_feature - p_a_when_second_feature)",
  "rationale": "This metric calculates the difference in the probability of choosing Option A between trials where Option A is supported by the highly diagnostic top feature (validity 0.95) versus trials where Option A is supported by the second-best feature (validity 0.75). According to the Strategy Mixture model, TTB strongly prefers A in both cases and Tallying strongly prefers B in both cases, leading to a similar mixture probability and a difference near zero. In contrast, Noisy TTB predicts a large difference because the 0.95 validity is robust to noise, whereas the 0.75 validity frequently swaps ranks with the closely clustered lower validities, leading to fewer choices of Option A in the second condition."
}
```

## Usage

```json
{
  "prompt_token_count": 3638,
  "candidates_token_count": 437,
  "total_token_count": 5991
}
```
