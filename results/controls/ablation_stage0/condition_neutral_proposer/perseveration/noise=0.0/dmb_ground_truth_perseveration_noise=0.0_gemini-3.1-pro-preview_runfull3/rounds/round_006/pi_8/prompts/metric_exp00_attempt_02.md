# metric_exp00_attempt_02

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
  trial 1: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 2: A=[1, 1, 0, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 3: A=[1, 0, 1, 1, 0]  B=[0, 1, 0, 0, 0]
  trial 4: A=[0, 1, 1, 1, 0]  B=[1, 0, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 6: A=[1, 1, 1, 0, 1]  B=[0, 0, 0, 1, 0]
  trial 7: A=[1, 1, 0, 1, 1]  B=[0, 0, 1, 0, 0]
  trial 8: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]

**Rationale:** Both theories predict extreme choice inertia (zero intra-subject variance), meaning a subject will press the same button for the entire experiment. However, Theory 1 determines this button by evaluating the unweighted tally of features on the *first* trial, whereas Theory 2 determines it via a pre-existing spatial bias, ignoring the stimulus completely. Because the trial order is randomized per subject, we cannot control which trial appears first. To force Theory 1 to make a consistent prediction, we design every trial such that Option A has a strictly higher unweighted sum of features than Option B. Consequently, no matter which trial appears first, Theory 1 will evaluate Option A as superior, choose it, and then deterministically repeat Option A for the rest of the experiment (predicting 100% Option A choices for all subjects). In contrast, Theory 2 predicts that subjects with a spatial bias for Option B will choose Option B on every trial, despite Option A being objectively superior. Observing subjects who exclusively choose Option B would strictly falsify Theory 1 and support Theory 2.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** First-Trial Feature Evaluation then Choice Inertia

**Parameters:**
- dummy: [0.0, 1.0]

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # Dummy parameter read to satisfy requirement
    dummy = float(parameters.get('dummy', 0.5))
    
    first_resp = None
    try:
        if history is not None:
            # Handle dict-of-lists format
            if isinstance(history, dict) and 'response' in history and len(history['response']) > 0:
                first_resp = history['response'][0]
            # Handle list-of-dicts or list-of-ints format just in case
            elif isinstance(history, (list, tuple)) or type(history).__name__ == 'ndarray':
                if len(history) > 0:
                    item = history[0]
                    if isinstance(item, dict) and 'response' in item:
                        first_resp = item['response']
                    elif isinstance(item, (int, float, np.integer)):
                        first_resp = int(item)
    except Exception:
        pass
        
    # Choice inertia: if we have a past response, repeat it entirely
    if first_resp is not None:
        if first_resp == 0:
            return np.array([1.0, 0.0])
        else:
            return np.array([0.0, 1.0])
            
    # First trial: unweighted tally of features
    try:
        a = np.sum(state['option_a_ratings'])
        b = np.sum(state['option_b_ratings'])
        if a > b:
            return np.array([1.0, 0.0])
        elif b > a:
            return np.array([0.0, 1.0])
    except Exception:
        pass
        
    # If all features tie or parsing fails, guess randomly
    return np.array([0.5, 0.5])
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
**Description:** Extreme Spatial Strategy / Extreme Position Bias: In the absence of correctness feedback and when confronted with complex, multi-cue choices, subjects completely disengage from the task. They adopt a degenerate strategy of pressing exactly one button (either always Option A or always Option B) for the entirety of the experiment. This stimulus-independent behavior ignores all feature values and previous history.

**Parameters:**
- preferred_side: {0, 1}

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    # The subject has a single preferred side for the entire experiment
    preferred_side = int(parameters['preferred_side'])
    
    # Predict exactly 1.0 for the preferred option and 0.0 for the other
    if preferred_side == 0:
        return np.array([1.0, 0.0])
    else:
        return np.array([0.0, 1.0])
```

**`policy source code`:**
```python
def policy(probs):
    import numpy as np
    probs = np.asarray(probs, dtype=np.float64)
    probs /= probs.sum()
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
[0] rationale: The advocated theory predicts that every subject will choose Option A (response=0) on the first trial because Option A always has more positive features, and then stick with it due to choice inertia. Thus, the mean response will be exactly 0 with zero between-subject variance. The competing theory predicts that subjects will randomly choose a preferred side and stick with it, resulting in half the subjects choosing Option B (response=1) consistently. The mean response will thus be ~0.5 with high between-subject variance, easily distinguished by a t-test.
metric_source:
def metric(data: pd.DataFrame) -> float:
    return float(data['response'].mean())
outcome: self_sim=0.3800 (var=0.2356) adversary_sim=0.5000 (var=0.2500) welch_t=-0.861 p=0.3935 (N=25, alpha=0.01) -> reject

[1] rationale: Theory 1 predicts that subjects evaluate the unweighted sum of features on their very first trial and deterministically choose the option with the higher sum. They then repeat this spatial choice (Left or Right) for the rest of the experiment due to choice inertia. Theory 2 predicts that subjects completely ignore the features and choose based on a pre-existing spatial bias from the very first trial. By isolating the response on the *first* trial and checking if it aligns with the higher-sum option, we expect a mean of 1.0 (with 0 variance) for Theory 1, and a mean of 0.5 (with 0.25 variance) for Theory 2, perfectly discriminating the two theories.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Extract the first trial for each subject
    first_trials = data.groupby('subject_id').first()
    
    # Calculate the sum of features for both options on the first trial
    sum_a = first_trials['option_a_ratings'].apply(sum)
    sum_b = first_trials['option_b_ratings'].apply(sum)
    
    # Determine which side has the higher sum (0 for Option A, 1 for Option B)
    higher_sum_side = (sum_b > sum_a).astype(int)
    
    # Check if the subject's response matches the side with the higher sum
    chose_higher = (first_trials['response'] == higher_sum_side).astype(float)
    
    return float(chose_higher.mean())
outcome: self_sim=0.6200 (var=0.2356) adversary_sim=0.5000 (var=0.2500) welch_t=+0.861 p=0.3935 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    data['sum_a'] = data['option_a_ratings'].apply(sum)\n    data['sum_b'] = data['option_b_ratings'].apply(sum)\n    data['higher_sum'] = (data['sum_b'] > data['sum_a']).astype(int)\n    return float((data['response'] == data['higher_sum']).mean())",
  "rationale": "Measures the overall proportion of trials where the subject's choice matches the option with the higher unweighted sum of features. If Theory 1's choice inertia fails or is overridden by the stimulus evaluation on subsequent trials, this match rate will be high. If Theory 2 is correct, subjects ignore the features entirely, resulting in a match rate of approximately 0.5."
}
```

## Usage

```json
{
  "prompt_token_count": 3647,
  "candidates_token_count": 193,
  "total_token_count": 31356
}
```
