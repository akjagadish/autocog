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
**Validities (n_features=5):** [0.95, 0.85, 0.6, 0.55, 0.5]

**Trial pairs (n=4):**
  trial 1: A=[1, 0, 0, 0, 0]  B=[0, 1, 0, 0, 0]
  trial 2: A=[1, 0, 1, 1, 1]  B=[0, 1, 0, 0, 0]
  trial 3: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 4: A=[1, 1, 0, 0, 0]  B=[1, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates 'Evidence Dilution and Non-linear Weighting Theory' (Advocated) from 'Sequential Search with Relative Evidence Thresholding' (Competing) by testing the impact of adding low-validity cues. Under the Advocated theory, adding weak cues to an option inflates the dilution denominator (sum of validities) while adding very little to the numerator (due to non-linear scaling), causing a paradoxical 'less-is-more' effect where adding cues hurts the option's choice probability. In contrast, the Competing theory evaluates cues sequentially and additively; weak cues either add to the accumulated evidence (increasing the option's preference) or are ignored due to early stopping, but they never penalize the option. Trial 1 establishes a baseline between the top two cues. Trial 2 adds weak cues to Option A, which the Advocated theory predicts will penalize A, while the Competing theory predicts will help or have no effect on A. Trial 4 equates the top two cues, leaving only weak cues favoring B; the Advocated theory predicts A is preferred because B is heavily diluted, whereas the Competing theory predicts B is chosen because it accumulates more evidence.

**Computed schedule:** 4 unique pairs × 24 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Evidence Dilution and Non-linear Weighting Theory (Validity-based Dilution with Amplified Penalty): Decision-makers evaluate options by integrating the validities of present features. However, instead of purely adding evidence, they partially average it. The presence of many low-validity features can paradoxically dilute the overall subjective value of an option (Evidence Dilution). This dilution is proportional to the sum of the validities of the present cues, and subjects apply a non-linear scaling to feature validities, amplifying the impact of the most valid cues. A potentially strong dilution penalty allows for severe subjective devaluation of options burdened with numerous weak features.

**Parameters:**
- lambda_val: [1.0, 20.0]
- gamma: [0.0, 10.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    lambda_val = float(parameters["lambda_val"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Non-linear weighting to capture TTB-like reliance on top cues
    w = val ** lambda_val
    
    # Dilute by the sum of validities of the present cues
    sum_val_a = np.sum(val * a)
    sum_val_b = np.sum(val * b)
    
    # Calculate subjective values with a dilution factor (gamma)
    v_a = np.sum(w * a) / (sum_val_a ** gamma) if sum_val_a > 0 else 0.0
    v_b = np.sum(w * b) / (sum_val_b ** gamma) if sum_val_b > 0 else 0.0
    
    scores = np.array([v_a, v_b])
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
**Description:** Sequential Search with Relative Evidence Thresholding: Decision-makers evaluate cues in descending order of validity. Each active cue adds to an evidence accumulator, but the subjective weight of each subsequent cue decays exponentially. Instead of a fixed absolute threshold for early stopping, subjects use a relative stopping rule: they stop searching when the accumulated evidence difference between options exceeds a threshold proportional to the maximum possible remaining evidence. This relative threshold seamlessly bridges Take-The-Best (stopping when remaining evidence cannot change the outcome) and Tallying (integrating all cues), explaining dynamic strategy shifts without relying on artificial cue penalties.

**Parameters:**
- decay: [0.0, 1.0]
- threshold: [0.0, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    val = np.asarray(parameters["validities"], dtype=float)
    
    a, b = stim[0], stim[1]
    
    decay = float(parameters["decay"])
    threshold = float(parameters["threshold"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by validity in descending order
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    acc_a = 0.0
    acc_b = 0.0
    
    # Evaluate cues sequentially
    for k, j in enumerate(cue_order):
        # Exponential decay of subjective weight based on cue rank
        weight = val[j] * (decay ** k)
        
        # Add weight for each active cue
        if a[j] == 1:
            acc_a += weight
        if b[j] == 1:
            acc_b += weight
        
        # Calculate the maximum possible remaining evidence
        remaining_evidence = sum(val[cue_order[m]] * (decay ** m) for m in range(k + 1, len(cue_order)))
        
        # Early stopping if the evidence difference reaches the relative threshold
        if abs(acc_a - acc_b) >= threshold * remaining_evidence:
            break
            
    scores = np.array([acc_a, acc_b])
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
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    sum_a = data['option_a_ratings'].apply(sum)\n    sum_b = data['option_b_ratings'].apply(sum)\n    \n    t2_mask = (sum_a == 4) & (sum_b == 1)\n    t4_mask = (sum_a == 2) & (sum_b == 5)\n    \n    p_a_t2 = 1.0 - data.loc[t2_mask, 'response'].mean() if t2_mask.any() else 0.5\n    p_a_t4 = 1.0 - data.loc[t4_mask, 'response'].mean() if t4_mask.any() else 0.5\n    \n    return float(p_a_t4 - p_a_t2)",
  "rationale": "This metric calculates the difference in the probability of choosing Option A between Trial 4 and Trial 2. Under the Advocated theory (Evidence Dilution), Option A is penalized in Trial 2 due to the addition of weak cues, while Option B is heavily penalized in Trial 4 for the same reason, leading to a higher preference for A in Trial 4 than in Trial 2 (positive difference). Under the Competing theory (Sequential Search), adding valid cues only increases an option's evidence, meaning Option A is strongly preferred in Trial 2 and Option B is strongly preferred in Trial 4, resulting in a lower preference for A in Trial 4 than in Trial 2 (negative difference). This opposing prediction creates a strong discriminative contrast."
}
```

## Usage

```json
{
  "prompt_token_count": 3371,
  "candidates_token_count": 377,
  "total_token_count": 5716
}
```
