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
**Validities (n_features=5):** [0.95, 0.85, 0.75, 0.65, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0]  B=[0, 0, 1, 1, 1]
  trial 3: A=[1, 1, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 4: A=[1, 1, 1, 1, 0]  B=[0, 0, 0, 0, 1]
  trial 5: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 6: A=[1, 1, 0, 0, 0]  B=[1, 0, 1, 1, 1]
  trial 7: A=[1, 1, 1, 1, 1]  B=[1, 0, 0, 0, 0]
  trial 8: A=[1, 1, 1, 0, 0]  B=[1, 1, 0, 1, 1]

**Rationale:** To quantitatively dissociate pure Take The Best (TTB) from TTB with Cue-Reading Errors, we hold the highest-validity discriminating cue constant while systematically varying the support of the remaining downstream cues. Pure TTB assumes perfect cue reading and strictly ignores all cues following the first discriminating cue, predicting a constant choice probability across these trials. In contrast, the Cue-Reading Errors model posits a non-zero probability of misreading cues. If a discriminating top cue is accidentally misread as tied (e.g., both 1 or both 0), the decision process continues to the downstream cues. Therefore, as the downstream cues shift from strongly favoring the alternative option to strongly favoring the TTB-chosen option, the error-prone model predicts an increasing choice probability for the TTB-chosen option, whereas pure TTB predicts a flat line.

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Take The Best (TTB) heuristic posits a lexicographic decision rule where individuals search through features in descending order of validity. They stop at the first feature that discriminates between the two options, choosing the option with the positive value on that feature. If no feature discriminates, they guess randomly. To account for empirical levels of noise, the choice is mixed with a lapse rate (epsilon) that can span up to 1.0 (pure guessing).

**Parameters:**
- epsilon: [0.0, 1.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    # Sort features in descending order of validity
    order = np.argsort(validities)[::-1]
    
    p_core = np.array([0.5, 0.5])
    for idx in order:
        if a[idx] > b[idx]:
            p_core = np.array([1.0, 0.0])
            break
        elif b[idx] > a[idx]:
            p_core = np.array([0.0, 1.0])
            break
            
    epsilon = float(parameters["epsilon"])
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


## COMPETING THEORY
**Description:** Take-The-Best with Cue-Reading Errors: Decision-makers fundamentally rely on a lexicographic heuristic (Take-The-Best), searching through features in strictly descending order of validity. However, they have a non-zero probability of misreading or misremembering individual cue values. This preserves the strict non-compensatory stopping rule while introducing variance that mimics compensatory behavior in certain environments.

**Parameters:**
- p_error: [0.0, 0.2]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    
    validities = np.asarray(parameters["validities"], dtype=float)
    p_error = float(parameters["p_error"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by true validities descending
    order = np.argsort(validities)[::-1]
    
    n_samples = 200
    p_a = 0.0
    
    for _ in range(n_samples):
        # Independent probability of misreading each cue
        err_a = np.random.rand(len(a)) < p_error
        err_b = np.random.rand(len(b)) < p_error
        
        # Apply errors (flip 0 to 1, and 1 to 0)
        a_noisy = np.abs(a - err_a)
        b_noisy = np.abs(b - err_b)
        
        for idx in order:
            if a_noisy[idx] > b_noisy[idx]:
                p_a += 1.0
                break
            elif b_noisy[idx] > a_noisy[idx]:
                break
        else:
            p_a += 0.5
            
    p_a /= n_samples
    p_core = np.array([p_a, 1.0 - p_a])
    
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
[0] rationale: By comparing Trial 5 and Trial 1, we isolate the effect of downstream cues when the most valid cue discriminates in favor of A. Pure Take The Best strictly ignores all cues after the first, so the probability of choosing A should be identical in both trials (difference of 0). The Cue-Reading Errors model, however, allows for the first cue to be accidentally misread as a tie. If this happens in Trial 1, the downstream cues strongly favor B, leading to a choice of B. If it happens in Trial 5, the downstream cues still favor A, leading to a choice of A. Thus, the Cue-Reading Errors model predicts a significantly higher probability of choosing A in Trial 5 compared to Trial 1, yielding a positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    a_sums = data['option_a_ratings'].apply(sum)
    b_sums = data['option_b_ratings'].apply(sum)
    
    # Trial 1: A=[1, 0, 0, 0, 0], B=[0, 1, 1, 1, 1]
    trial_1_mask = (a_sums == 1) & (b_sums == 4)
    # Trial 5: A=[1, 1, 1, 1, 1], B=[0, 0, 0, 0, 0]
    trial_5_mask = (a_sums == 5) & (b_sums == 0)
    
    # response == 0 means choosing Option A. 
    # 1.0 - mean(response) gives the proportion of times Option A was chosen.
    p_a_trial_1 = 1.0 - data.loc[trial_1_mask, 'response'].mean()
    p_a_trial_5 = 1.0 - data.loc[trial_5_mask, 'response'].mean()
    
    return float(p_a_trial_5 - p_a_trial_1)
outcome: self_sim=0.0450 (var=0.0153) adversary_sim=0.1117 (var=0.0341) welch_t=-1.500 p=0.1411 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    a_sums = data['option_a_ratings'].apply(sum)\n    b_sums = data['option_b_ratings'].apply(sum)\n    \n    # Trials 1-5 all have a_sums + b_sums == 5\n    # Trials 6-8 have sums >= 6\n    mask_1_to_5 = (a_sums + b_sums == 5)\n    \n    # Group 1: Trials 1 and 2 (downstream cues strongly favor B)\n    g1_mask = mask_1_to_5 & (a_sums <= 2)\n    # Group 2: Trials 4 and 5 (downstream cues strongly favor A)\n    g2_mask = mask_1_to_5 & (a_sums >= 4)\n    \n    p_a_g1 = 1.0 - data.loc[g1_mask, 'response'].mean()\n    p_a_g2 = 1.0 - data.loc[g2_mask, 'response'].mean()\n    \n    return float(p_a_g2 - p_a_g1)",
  "rationale": "To reduce the within-subject variance of the metric, we pool trials with similar downstream structures rather than relying on a single pair of trials. Trials 1 to 5 share the property that the most valid cue discriminates in favor of A, while the sum of A and B's cues equals 5. By pooling Trials 1 and 2 (where downstream cues strongly favor B) and comparing them against Trials 4 and 5 (where downstream cues strongly favor A), we double the number of observations per subject compared to the previous attempt. Pure Take-The-Best ignores downstream cues and predicts a flat choice probability (difference ~ 0), whereas the Cue-Reading Errors model predicts a significant positive difference due to accidental misreading of the first cue."
}
```

## Usage

```json
{
  "prompt_token_count": 3673,
  "candidates_token_count": 450,
  "total_token_count": 6527
}
```
