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
**Validities (n_features=6):** [0.95, 0.9, 0.8, 0.7, 0.6, 0.55]

**Trial pairs (n=8):**
  trial 1: A=[1, 1, 0, 0, 0, 0]  B=[0, 0, 1, 1, 1, 1]
  trial 2: A=[1, 1, 0, 0, 0, 1]  B=[0, 0, 1, 1, 1, 0]
  trial 3: A=[1, 1, 0, 0, 1, 1]  B=[0, 0, 1, 1, 0, 0]
  trial 4: A=[1, 1, 0, 1, 1, 1]  B=[0, 0, 1, 0, 0, 0]
  trial 5: A=[1, 1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0, 0]
  trial 6: A=[1, 0, 1, 1, 0, 0]  B=[0, 1, 0, 0, 1, 1]
  trial 7: A=[1, 0, 1, 0, 0, 0]  B=[0, 1, 0, 1, 1, 1]
  trial 8: A=[1, 0, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1, 1]

**Rationale:** This design quantitatively dissociates Weighted Additive (WADD) theory from Sequential Cue Accumulation (SCA / Take-The-Best with Verification) by probing for 'early stopping' behavior. SCA assumes decision-makers accumulate evidence sequentially by validity and stop searching as soon as a threshold is reached. If the first few high-validity cues strongly favor one option and exceed this threshold, SCA completely ignores all remaining lower-validity cues, predicting identical choice probabilities across trials regardless of how the remaining cues are distributed. WADD, by contrast, integrates all information. It predicts that choice probabilities will systematically shift as the distribution of lower-validity cues changes, even when the top cues provide overwhelming evidence. By fixing the top two highly valid cues to favor Option A and systematically varying whether the remaining four cues favor A or B, we can cleanly test whether lower-validity cues are ignored (SCA) or integrated (WADD).

**Computed schedule:** 8 unique pairs × 12 reps = 96 trials per subject.



## ADVOCATED THEORY
**Description:** Weighted Additive (WADD) theory with non-linear scaling posits that decision-makers integrate all available information by weighting each feature by its validity, but subjectively scale these validities to adjust the degree of compensation. By exponentiating validities by a parameter gamma and then normalizing them, the model flexibly interpolates between equal weighting (Tallying), proportional weighting, and highly non-compensatory weighting (Take The Best) while keeping the integrated scores on a stable scale. Choices are made probabilistically using a softmax over the options' scores.

**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- gamma: [0.1, 30.0]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError("WADD expects a (2, n_features) stimulus.")
        
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    gamma = float(parameters["gamma"])
    
    # Non-linear scaling of validities with normalization to prevent score shrinkage
    weights = val ** gamma
    weights = weights / np.sum(weights)
    
    # Calculate overall scores by weighting features by their scaled validity
    a_score = np.dot(a, weights)
    b_score = np.dot(b, weights)
    scores = np.array([a_score, b_score])
    
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Numerically stable softmax
    z = beta * (scores - scores.max())
    e = np.exp(z)
    p_core = e / e.sum()
    
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
**Description:** Sequential Cue Accumulation (Take-The-Best with Verification): Decision-makers process cues sequentially in order of validity, accumulating evidence for or against each option. They stop searching when the absolute accumulated evidence reaches a threshold. If the evidence reaches the threshold, a decision is made based on the accumulated evidence. However, if all cues are exhausted and the accumulated evidence fails to reach the threshold (e.g., due to highly contradictory cues), the decision-maker defaults to guessing. This captures the chance-like behavior observed in some compensatory settings while maintaining Take-The-Best behavior when early cues provide overwhelming evidence.

**Parameters:**
- theta: [0.0, 1.0]
- gamma: [0.0, 5.0]
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    import numpy as np
    
    stim = np.asarray(stimulus, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    theta = float(parameters["theta"])
    gamma = float(parameters["gamma"])
    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])
    
    # Scale and normalize weights
    weights = val ** gamma
    weights = weights / np.sum(weights)
    
    # Search in order of descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    E = 0.0
    threshold_reached = False
    for j in cue_order:
        diff = a[j] - b[j]
        if diff != 0:
            E += diff * weights[j]
            # Stop if accumulated evidence reaches the threshold
            if abs(E) >= theta:
                threshold_reached = True
                break
            
    if not threshold_reached:
        p = np.array([0.5, 0.5])
    else:
        scores = np.array([E, -E])
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
[0] rationale: This metric calculates the difference in the probability of choosing Option A between Trial 4 and Trial 1. In both trials, the two most valid cues strongly favor Option A. However, Trial 1 has all remaining cues favoring Option B, whereas Trial 4 has almost all remaining cues favoring Option A. Weighted Additive (WADD) theory predicts that these lower-validity cues will be integrated, leading to a significantly higher probability of choosing A in Trial 4 than in Trial 1. Sequential Cue Accumulation (SCA), on the other hand, will likely hit its evidence threshold early on the first two cues and ignore the rest, predicting a difference close to zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    # Identify trial types by option A's ratings
    t_keys = data['option_a_ratings'].apply(tuple)
    
    t1 = (1, 1, 0, 0, 0, 0)
    t4 = (1, 1, 0, 1, 1, 1)
    
    # Mean response is the proportion of times Option B was chosen (response == 1)
    # We want the difference in P(Choose A) between Trial 4 and Trial 1
    # P(Choose A) = 1 - mean(response)
    
    mask_t1 = t_keys == t1
    mask_t4 = t_keys == t4
    
    if not mask_t1.any() or not mask_t4.any():
        return 0.0
        
    p_a_t1 = 1.0 - data[mask_t1]['response'].mean()
    p_a_t4 = 1.0 - data[mask_t4]['response'].mean()
    
    return float(p_a_t4 - p_a_t1)
outcome: self_sim=0.0550 (var=0.0557) adversary_sim=0.0167 (var=0.0400) welch_t=+0.619 p=0.5386 (N=25, alpha=0.01) -> reject

[1] rationale: By comparing Trial 6 and Trial 8, we avoid the 'ceiling effect' present in trials where the top two validities both point to the same option. In Trials 6 and 8, the highest validity cue favors Option A, but the second highest validity cue favors Option B, largely canceling each other out. For WADD, this cancellation allows the lower-validity cues to drive the decision, resulting in a significantly higher probability of choosing A in Trial 6 than in Trial 8. For SCA, the behavior is strongly dichotomous: either the first cue's evidence alone exceeds the threshold (leading to identical choices for A in both trials), or the conflicting top cues keep the accumulated evidence below the threshold, leading to random guessing in both trials. In either case, SCA predicts a much smaller difference between these two trials than WADD.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    
    # Convert lists to tuples to make them hashable for comparison
    t_keys = data['option_a_ratings'].apply(tuple)
    
    # Trial 6: Top cues conflict, lower cues slightly favor A
    t6 = (1, 0, 1, 1, 0, 0)
    # Trial 8: Top cues conflict, lower cues strongly favor B
    t8 = (1, 0, 0, 0, 0, 0)
    
    mask_t6 = t_keys == t6
    mask_t8 = t_keys == t8
    
    if not mask_t6.any() or not mask_t8.any():
        return 0.0
        
    # P(Choose A) is 1.0 minus the mean of responses (where 1 indicates choosing B)
    p_a_t6 = 1.0 - data[mask_t6]['response'].mean()
    p_a_t8 = 1.0 - data[mask_t8]['response'].mean()
    
    # Return the difference in probability of choosing A
    return float(p_a_t6 - p_a_t8)
outcome: self_sim=0.1150 (var=0.0608) adversary_sim=0.0467 (var=0.0520) welch_t=+1.017 p=0.3141 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    import pandas as pd\n    # Convert ratings to hashable tuples\n    t_keys = data['option_a_ratings'].apply(tuple)\n    \n    # Focus on the highly conflicting trials where SCA is likely to fail to reach its evidence threshold\n    t6 = (1, 0, 1, 1, 0, 0)\n    t7 = (1, 0, 1, 0, 0, 0)\n    t8 = (1, 0, 0, 0, 0, 0)\n    \n    mask = t_keys.isin([t6, t7, t8])\n    if not mask.any():\n        return 0.0\n        \n    # Calculate the mean response for each of these trial types\n    subset = data[mask]\n    t_keys_subset = t_keys[mask]\n    means = subset.groupby(t_keys_subset)['response'].mean()\n    \n    # Calculate 'determinism': how far the mean response is from 0.5 (random guessing)\n    determinism = (means - 0.5).abs().mean()\n    \n    return float(determinism)",
  "rationale": "This metric calculates the 'determinism' of choices on the most highly conflicting trials (Trials 6, 7, and 8). In these trials, the highest validity cue favors Option A, but the next highest validity cue favors Option B, and the remaining cues are distributed differently. Sequential Cue Accumulation (SCA) processes these cues sequentially. Because the cues alternate in supporting A and B, the accumulated evidence constantly fluctuates around zero and is very likely to never reach the decision threshold, causing the SCA model to default to random guessing (P = 0.5). Weighted Additive (WADD) theory, however, integrates all cues into a single compensatory score. Depending on the subjective scaling (gamma), one option will consistently outscore the other, leading to highly deterministic choices (P near 0 or 1, subject only to epsilon noise). By measuring the absolute deviation of the choice proportion from 0.5 on these specific trials, we can cleanly separate the guessing behavior of SCA from the deterministic integration of WADD, while keeping between-subject variance low."
}
```

## Usage

```json
{
  "prompt_token_count": 4450,
  "candidates_token_count": 524,
  "total_token_count": 12041
}
```
