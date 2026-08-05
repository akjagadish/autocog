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
  trial 1: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 0]
  trial 2: A=[0, 1, 0, 1, 0]  B=[1, 0, 1, 0, 0]
  trial 3: A=[0, 1, 1, 0, 0]  B=[1, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0]
  trial 5: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]
  trial 6: A=[0, 0, 1, 0, 0]  B=[1, 1, 0, 0, 0]
  trial 7: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 0, 0]
  trial 8: A=[0, 1, 0, 0, 0]  B=[0, 0, 1, 0, 0]

**Rationale:** To quantitatively dissociate Pure Tallying from Hybrid Tallying, we focus on the specific conditions where their predictions diverge. Pure Tallying predicts a strict 50/50 guess whenever the number of winning features is equal, completely ignoring cue validities. Hybrid Tallying, however, applies a 'premium' to the single most valid cue, predicting a preference for the option that wins this top cue even when the overall tally is tied. Furthermore, when one option wins the tally by exactly one cue but loses the top cue, Pure Tallying predicts a standard preference strength based on a score difference of 1, whereas Hybrid Tallying predicts a significantly attenuated preference (score difference of 1 - premium). This design includes tally ties with top-cue wins, narrow tally advantages opposing the top cue, and control trials where the top cue is tied, providing a comprehensive test of whether the top cue exerts any special influence over choices.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** People compare two options by counting, across all features, how often one option has a higher value than the other. The option that wins on more features is chosen. Tallying discards cardinal magnitudes — only the sign of each feature-wise comparison matters — so the heuristic is robust to monotone rescaling of individual features and cannot be swayed by a single large feature difference in the way Equal-Weight can. Ties on an individual feature contribute nothing to either count: that cue is simply treated as uninformative for the pair. No feature is privileged, in contrast to Take The Best; every cue contributes equally to the tally. When the two counts are equal the heuristic has no basis for preference and the learner must guess. Response noise enters through a softmax over the two tallies with inverse temperature beta (interpolating between fully deterministic choice at large beta and uniform guessing at beta = 0), plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Tallying heuristic (Dawes 1979; Gigerenzer &
    # Goldstein 1999). Stimulus is the pair of option feature vectors
    # for the current trial: array-like of shape (2, n_features),
    # with row 0 = option A, row 1 = option B. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"Tallying expects a (2, n_features) stimulus; got shape {stim.shape}."
        )

    a, b = stim[0], stim[1]
    # Count strict feature-wise wins; ties contribute to neither option.
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    scores = np.array([a_wins, b_wins])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. When
    # a_wins == b_wins the softmax is exactly uniform regardless of
    # beta, which is the correct behavior for an undiscriminating
    # tally.
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()

    n_opts = p_core.shape[0]
    return (1.0 - epsilon) * p_core + epsilon * (np.ones(n_opts) / n_opts)
```

**`policy source code`:**
```python
def policy(probabilities):
    probabilities = probabilities / probabilities.sum()  # Ensure valid probabilities.
    return np.random.choice(len(probabilities), p=probabilities)
```


## COMPETING THEORY
**Description:** Hybrid Tallying: Decision-makers primarily rely on a frugal Tallying heuristic, counting the number of features where one option dominates another. However, they are not completely blind to cue validities; they give a slight premium to the single most valid cue. This premium acts strictly as a tie-breaker or a soft modulator and is never large enough to overcome a strict difference in tallies, blending the robustness of Tallying with a minimal sensitivity to cue hierarchy.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- premium: [0.0, 0.99]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Hybrid Tallying expects a (2, n_features) stimulus.")

    val = np.asarray(parameters["validities"], dtype=float)
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    premium = float(parameters["premium"])
    
    a, b = stim[0], stim[1]
    
    # Pure tallying component
    a_wins = float(np.sum(a > b))
    b_wins = float(np.sum(b > a))
    
    # Identify the most valid cue (stable sort to break ties by original index)
    top_cue = np.argsort(-val, kind="stable")[0]
    
    # Premium for winning the most valid cue
    a_top = float(a[top_cue] > b[top_cue])
    b_top = float(b[top_cue] > a[top_cue])
    
    score_a = a_wins + premium * a_top
    score_b = b_wins + premium * b_top
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import numpy as np\n    \n    a_ratings = np.stack(data['option_a_ratings'].values)\n    b_ratings = np.stack(data['option_b_ratings'].values)\n    \n    a_wins = (a_ratings > b_ratings).sum(axis=1)\n    b_wins = (b_ratings > a_ratings).sum(axis=1)\n    \n    a_top = a_ratings[:, 0] > b_ratings[:, 0]\n    b_top = b_ratings[:, 0] > a_ratings[:, 0]\n    \n    is_tie = (a_wins == b_wins)\n    \n    target_trials = is_tie & (a_top | b_top)\n    \n    if not np.any(target_trials):\n        return 0.5\n        \n    responses = data['response'].values[target_trials]\n    a_top_target = a_top[target_trials]\n    b_top_target = b_top[target_trials]\n    \n    match = ( (responses == 0) & a_top_target ) | ( (responses == 1) & b_top_target )\n    \n    return float(np.mean(match))",
  "rationale": "This metric calculates the proportion of choices that align with the single most valid cue, strictly on trials where the overall tally is tied. Pure Tallying predicts a 50/50 guess on these trials since it ignores cue validities, resulting in an expected value of 0.5. Hybrid Tallying predicts a value significantly greater than 0.5 because it uses the top cue as a tie-breaker."
}
```

## Usage

```json
{
  "prompt_token_count": 3467,
  "candidates_token_count": 417,
  "total_token_count": 5860
}
```
