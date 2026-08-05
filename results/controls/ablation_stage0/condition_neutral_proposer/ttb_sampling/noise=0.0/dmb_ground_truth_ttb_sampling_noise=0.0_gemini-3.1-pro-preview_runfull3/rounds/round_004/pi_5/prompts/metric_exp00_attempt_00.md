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
**Validities (n_features=6):** [0.95, 0.75, 0.65, 0.58, 0.54, 0.51]

**Trial pairs (n=10):**
  trial 1: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]
  trial 2: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 3: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 1]
  trial 4: A=[0, 0, 0, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 5: A=[0, 1, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 6: A=[0, 0, 1, 1, 0, 0]  B=[0, 0, 0, 0, 1, 1]
  trial 7: A=[0, 0, 0, 1, 1, 0]  B=[0, 0, 0, 0, 0, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 0, 0, 0]
  trial 9: A=[0, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 0, 0]
  trial 10: A=[0, 0, 1, 0, 0, 0]  B=[0, 0, 0, 1, 1, 0]

**Rationale:** To maximally distinguish Theory 1 (Two-Stage Confidence-Threshold Strategy Selection) from Theory 2 (pure Take-The-Best), we exploit the confidence threshold mechanism in Theory 1. Theory 1 uses TTB if the maximum discriminating validity is above a subjective threshold (between 0.5 and 0.8), but falls back to a compensatory Weighted Additive (WADD) rule if it falls below. Theory 2 always uses TTB. We use 6 features with validities ranging from 0.95 to 0.51. We design critical dissociation trials where the highest discriminating validity is systematically varied (0.95, 0.75, 0.65, 0.58). In these trials, the top discriminating cue favors Option A (which TTB will pick), while the sum of the remaining lower-validity cues favors Option B (which WADD will pick). For the 0.95 cue, both theories will confidently use TTB and agree on Option A. However, for the 0.75, 0.65, and 0.58 cues, Theory 1 may fall below its confidence threshold and switch to WADD (picking Option B), while Theory 2 will continue to pick Option A. By observing at which validity level the choice switches from A to B, we can precisely identify the presence and value of the confidence threshold.

**Computed schedule:** 10 unique pairs × 9 reps = 90 trials per subject.



## ADVOCATED THEORY
**Description:** Two-Stage Confidence-Threshold Strategy Selection: Decision-makers default to the fast and frugal Take-The-Best (TTB) heuristic, evaluating options based solely on the most valid discriminating cue. However, if the validity of this top discriminating cue falls below a subjective confidence threshold, the decision-maker deems the single-cue evidence insufficient and falls back to a compensatory Weighted Additive (WADD) strategy that integrates all available cues.

**Parameters:**
- confidence_threshold: [0.5, 0.8]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("Stimulus must be shape (2, n_features).")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    confidence_threshold = float(parameters["confidence_threshold"])
    
    diff = stim[0] - stim[1]
    discrim_mask = diff != 0
    
    scores = np.zeros(2)
    if np.any(discrim_mask):
        discrim_validities = validities[discrim_mask]
        max_v = np.max(discrim_validities)
        
        if max_v >= confidence_threshold:
            # Strategy 1: Take-The-Best (TTB)
            top_idx = np.where((discrim_mask) & (validities == max_v))[0][0]
            if stim[0, top_idx] > stim[1, top_idx]:
                scores[0] = 1.0
            else:
                scores[1] = 1.0
        else:
            # Strategy 2: Weighted Additive (WADD) fallback
            wadd_scores = stim @ validities
            if wadd_scores[0] > wadd_scores[1]:
                scores[0] = 1.0
            elif wadd_scores[1] > wadd_scores[0]:
                scores[1] = 1.0
                
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
    return int(np.random.choice(len(probabilities), p=probabilities))
```


## COMPETING THEORY
**Description:** People compare two options using the Take-The-Best (TTB) heuristic. TTB is a fast-and-frugal lexicographic strategy that searches through cues in descending order of their validity. The first cue that discriminates between the two options completely determines the choice, and all remaining lower-validity cues are ignored. If no cue discriminates, the decision maker guesses randomly. Response noise enters through a softmax over the binary TTB outcome with inverse temperature beta, plus an independent lapse rate epsilon.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("TTB expects a (2, n_features) stimulus.")
        
    validities = np.asarray(parameters["validities"], dtype=float)
    
    # Sort cues by validity in descending order
    order = np.argsort(validities)[::-1]
    
    scores = np.zeros(2)
    for idx in order:
        if stim[0, idx] > stim[1, idx]:
            scores[0] = 1.0
            break
        elif stim[1, idx] > stim[0, idx]:
            scores[1] = 1.0
            break
            
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Identify trials where option A is favored only by a single cue (which is not the most valid cue overall)\n    # These are trials where TTB picks A, but WADD picks B, and the max validity is <= 0.75.\n    is_target = data['option_a_ratings'].apply(lambda x: sum(x) == 1 and x[0] == 0)\n    if not is_target.any():\n        return 0.0\n    return float(data.loc[is_target, 'response'].mean())",
  "rationale": "This metric isolates trials where Option A is supported by exactly one cue (which is not the highest validity cue of 0.95), while Option B is supported by multiple lower-validity cues such that the WADD score favors B. Theory 2 (pure TTB) will consistently choose Option A (response = 0). Theory 1, however, will often evaluate the top cue's validity (<= 0.75) as falling below its confidence threshold and switch to WADD, thereby choosing Option B (response = 1). Thus, the mean response on these target trials will be significantly higher for Theory 1 than for Theory 2."
}
```

## Usage

```json
{
  "prompt_token_count": 3631,
  "candidates_token_count": 292,
  "total_token_count": 7440
}
```
