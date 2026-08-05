# metric_exp01_attempt_01

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
**Validities (n_features=5):** [0.9, 0.8, 0.65, 0.6, 0.55]

**Trial pairs (n=5):**
  trial 1: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 1]
  trial 2: A=[0, 0, 1, 0, 0]  B=[0, 0, 0, 1, 0]
  trial 3: A=[0, 0, 1, 1, 1]  B=[0, 0, 0, 0, 0]
  trial 4: A=[1, 0, 0, 0, 0]  B=[0, 1, 1, 1, 1]
  trial 5: A=[1, 1, 1, 1, 1]  B=[0, 0, 0, 0, 0]

**Rationale:** To quantitatively dissociate the Validity-Threshold with WADD Fallback model (Advocated Theory) from the pure Take The Best model (Competing Theory), we systematically vary the validity of the top discriminating cue and the support from lower-validity cues. Pure TTB strictly stops at the first discriminating cue, ignoring its validity magnitude and all subsequent cues, and thus predicts an identical, flat probability of choosing Option A across all trials where the top cue favors A. In contrast, the Advocated Theory posits a validity threshold: if the top discriminating cue's validity is below the threshold, the decision-maker falls back to a compensatory WADD strategy. By setting the top discriminating cue to lower validities (e.g., Feature 3) and parametrically shifting the remaining cues from strongly favoring Option B to strongly favoring Option A, the Advocated Theory predicts a monotonic shift in choice probability (due to WADD fallback), directly contradicting the perfectly flat line predicted by pure TTB. We also include trials where Feature 1 discriminates to demonstrate that both models agree when the top cue's validity exceeds the threshold.

**Computed schedule:** 5 unique pairs × 19 reps = 95 trials per subject.



## ADVOCATED THEORY
**Description:** Validity-Threshold Model with WADD Fallback: Decision-makers process cues sequentially in order of validity, but their choice of strategy depends on the absolute strength of the best available evidence. If the first discriminating cue has a high validity (exceeding an internal threshold), the decision-maker relies purely on Take The Best (TTB), ignoring all other cues. If the top discriminating cue falls below this threshold, the decision-maker falls back to a compensatory Weighted Additive (WADD) strategy, integrating all available cues weighted by their validities.

**Parameters:**
- validity_threshold: [0.0, 1.0]
- beta_ttb: [0.1, 20.0]
- beta_comp: [0.1, 10.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, state, history):
    import numpy as np
    
    stim = np.asarray(state, dtype=float)
    a, b = stim[0], stim[1]
    val = np.asarray(parameters["validities"], dtype=float)
    
    threshold = float(parameters["validity_threshold"])
    beta_ttb = float(parameters["beta_ttb"])
    beta_comp = float(parameters["beta_comp"])
    epsilon = float(parameters["epsilon"])
    
    # Sort cues by descending validity
    cue_order = np.argsort(-val, kind="stable").tolist()
    
    disc_cue_idx = -1
    for j in cue_order:
        if a[j] != b[j]:
            disc_cue_idx = j
            break
            
    if disc_cue_idx == -1:
        return np.array([0.5, 0.5])
        
    v_disc = val[disc_cue_idx]
    
    if v_disc >= threshold:
        # Take The Best: rely solely on the first discriminating cue
        scores = np.array([1.0, 0.0]) if a[disc_cue_idx] > b[disc_cue_idx] else np.array([0.0, 1.0])
        z = beta_ttb * scores
    else:
        # WADD: weighted sum of cues as a compensatory fallback
        wadd_a = float(np.sum(val * a))
        wadd_b = float(np.sum(val * b))
        scores = np.array([wadd_a, wadd_b])
        z = beta_comp * scores
        
    z -= np.max(z)  # Numerical stability
    p_core = np.exp(z) / np.sum(np.exp(z))
    
    return (1.0 - epsilon) * p_core + epsilon * np.array([0.5, 0.5])
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
**Description:** People compare two options by consulting cues one at a time in order of validity, stopping at the first cue that discriminates between the two options. That cue alone determines the choice: the option with the higher value on the discriminating cue wins, and no other cue is consulted. Cues with lower validity are never reached when a higher-validity cue already discriminates, so Take The Best is a "one-reason" decision rule — only a single feature is ever used on any given choice. Because only the sign of the comparison on the top discriminating cue matters, TTB ignores both the magnitude of that difference and all information on lower-validity cues, making it maximally frugal in the use of evidence. Cue validities are subjective and can be learned, inferred from instructions, or fixed by task structure. When no cue discriminates (all feature-wise comparisons tie) the learner has no basis for preference and must guess. Response noise enters through a softmax over the binary TTB score (winner = 1, loser = 0) with inverse temperature beta, plus an independent lapse that with probability epsilon replaces the softmax output with a uniform pick over the two options.


**Parameters:**
- beta: [0.1, 20.0]
- epsilon: [0.0, 0.5]
- validities: validities

**`predict source code`:**
```python
def predict(parameters, stimulus, history):
    # Paper-faithful Take The Best (Gigerenzer & Goldstein 1996).
    # Stimulus is the pair of option feature vectors for the current
    # trial: array-like of shape (2, n_features), row 0 = option A,
    # row 1 = option B. Cue cascade: features are consulted in order
    # of descending validity; the first discriminating cue (strict
    # inequality) determines the winner; if no cue discriminates,
    # the model guesses uniformly. History is ignored.
    stim = np.asarray(stimulus, dtype=float)
    if stim.ndim != 2 or stim.shape[0] != 2:
        raise ValueError(
            f"TTB expects a (2, n_features) stimulus; got shape {stim.shape}."
        )
    n_features = stim.shape[1]

    val = np.asarray(parameters["validities"], dtype=float)
    if val.shape[0] != n_features:
        raise ValueError(
            f"validities length {val.shape[0]} != n_features {n_features}."
        )
    # Descending validity; argsort is stable so validity ties break
    # toward the earlier feature index.
    cue_order = np.argsort(-val, kind="stable").tolist()

    a, b = stim[0], stim[1]
    winner = None
    for j in cue_order:
        if a[j] > b[j]:
            winner = 0
            break
        if b[j] > a[j]:
            winner = 1
            break

    if winner is None:
        # No discriminating cue — pure guess.
        return np.ones(2) / 2.0

    scores = np.array([1.0, 0.0]) if winner == 0 else np.array([0.0, 1.0])

    beta = float(parameters["beta"])
    epsilon = float(parameters["epsilon"])

    # Softmax with max-subtraction for numerical stability. For the
    # binary TTB score this collapses to sigmoid(beta) for the winner,
    # giving a direct mapping from beta onto the paper's flip-noise
    # levels (beta=0 ↔ 50/50; beta ≫ 1 ↔ deterministic).
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
[0] rationale: In all 6 trials of the experimental design, the top discriminating cue favors Option A. Thus, pure Take The Best (TTB) predicts a constant probability of choosing A across all trials (ignoring all lower-validity cues). However, the Advocated Theory posits a fallback to the Weighted Additive (WADD) strategy when the top cue's validity falls below a threshold. In trials where WADD favors Option B (which corresponds to trials where the sum of features for B is greater than A), the probability of choosing A will drop. This metric computes the difference in the rate of choosing A between trials where WADD favors A and trials where WADD favors B. TTB predicts this difference will be near 0, whereas the Advocated Theory predicts a positive difference.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate sum of ratings for A and B
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    # Trials where WADD favors A vs B
    wadd_favors_a = sum_a >= sum_b
    
    # Proportion of choosing A (response == 0)
    p_a_given_wadd_a = (data.loc[wadd_favors_a, 'response'] == 0).mean()
    p_a_given_wadd_b = (data.loc[~wadd_favors_a, 'response'] == 0).mean()
    
    return float(p_a_given_wadd_a - p_a_given_wadd_b)
outcome: self_sim=0.0616 (var=0.0254) adversary_sim=0.0125 (var=0.0055) welch_t=+1.397 p=0.1714 (N=25, alpha=0.01) -> reject

[1] rationale: In Trial 1, the first discriminating cue has a very high validity (0.95), making it extremely likely to exceed the threshold in the Advocated Theory, leading to a TTB strategy (which favors A). In Trial 6, the first discriminating cue has a low validity (0.65), making it much more likely to fall below the threshold and trigger the WADD fallback strategy (which favors B). Thus, the Advocated Theory predicts a large drop in the probability of choosing A from Trial 1 to Trial 6. In contrast, pure TTB ignores both the validity magnitude and the lower-validity cues, predicting a perfectly flat probability of choosing A across both trials.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Identify trials by the sum of features to distinguish them
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trial 1: Top cue is Feature 1 (validity 0.95). TTB favors A, WADD favors B.
    trial_1_mask = (sum_a == 1) & (sum_b == 5)
    
    # Trial 6: Top cue is Feature 4 (validity 0.65). TTB favors A, WADD favors B.
    trial_6_mask = (sum_a == 4) & (sum_b == 5)
    
    # Probability of choosing Option A (response == 0)
    p_a_t1 = (data.loc[trial_1_mask, 'response'] == 0).mean()
    p_a_t6 = (data.loc[trial_6_mask, 'response'] == 0).mean()
    
    # Pure TTB predicts this difference will be 0, because the top cue favors A in both.
    # The Advocated Theory predicts a positive difference because Feature 1's high validity 
    # triggers TTB (choosing A), while Feature 4's lower validity triggers WADD fallback (choosing B).
    return float(p_a_t1 - p_a_t6)
outcome: self_sim=0.0837 (var=0.0763) adversary_sim=0.0250 (var=0.0217) welch_t=+0.938 p=0.3544 (N=25, alpha=0.01) -> reject

[2] rationale: To reduce per-subject variance and maximize the contrast between the theories, this metric compares the probability of choosing Option A across two pooled sets of trials. In Trials 4 and 5, both the Take The Best (TTB) and Weighted Additive (WADD) strategies favor Option A, leading to a high probability of choosing A under both models. In Trials 3 and 6, TTB still favors A, but WADD strongly favors B. Because the top discriminating cues in Trials 3 and 6 have relatively low validities (0.75 and 0.65), the Advocated Theory predicts these cues will often fall below the subject's validity threshold, triggering the WADD fallback and dropping the probability of choosing A. Pure TTB, ignoring validity magnitudes and lower-validity cues, predicts a flat, high probability of choosing A across all these trials. By pooling two trials per condition, we reduce response noise and stabilize the per-subject metric, yielding a large positive difference for the Advocated Theory and a near-zero difference for TTB.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import numpy as np
    
    # Calculate the sum of ratings for options A and B to uniquely identify trials
    sum_a = data['option_a_ratings'].apply(sum)
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trials 4 and 5: Both TTB and WADD favor Option A.
    # T4: sum_a=4, sum_b=4. T5: sum_a=6, sum_b=2.
    mask_wadd_a = ((sum_a == 4) & (sum_b == 4)) | ((sum_a == 6) & (sum_b == 2))
    
    # Trials 3 and 6: TTB favors Option A, but WADD strongly favors Option B.
    # Moreover, the top discriminating cues in these trials have lower validities (0.75 and 0.65),
    # making them highly likely to fall below the Validity-Threshold, triggering WADD fallback.
    # T3: sum_a=3, sum_b=5. T6: sum_a=4, sum_b=5.
    mask_wadd_b = ((sum_a == 3) & (sum_b == 5)) | ((sum_a == 4) & (sum_b == 5))
    
    # Proportion of choosing A (response == 0)
    p_a_wadd_a = (data.loc[mask_wadd_a, 'response'] == 0).mean()
    p_a_wadd_b = (data.loc[mask_wadd_b, 'response'] == 0).mean()
    
    return float(p_a_wadd_a - p_a_wadd_b)
outcome: self_sim=0.0988 (var=0.0507) adversary_sim=0.0169 (var=0.0088) welch_t=+1.678 p=0.1031 (N=25, alpha=0.01) -> reject

[3] rationale: To maximize the contrast between pure Take The Best (TTB) and the Validity-Threshold with WADD Fallback model, we isolate Trials 3 and 5. In both trials, the top discriminating cue is the exact same (Feature 3, validity 0.75) and favors Option A. Pure TTB strictly follows this top cue, predicting identical probabilities of choosing A across both trials. However, if the subject's validity threshold exceeds 0.75, the Advocated Theory predicts a fallback to the compensatory WADD strategy. In Trial 5, WADD strongly favors Option A, but in Trial 3, WADD strongly favors Option B. By comparing the choice rates specifically on these two extreme trials, we avoid diluting the effect with weaker trials (like Trials 4 and 6), thereby maximizing the mean difference for the Advocated Theory while keeping TTB's prediction exactly at zero.
metric_source:
def metric(data: pd.DataFrame) -> float:
    import pandas as pd
    import numpy as np
    
    # Calculate the sum of ratings to uniquely identify trials
    sum_a = data['option_a_ratings'].apply(np.sum)
    sum_b = data['option_b_ratings'].apply(np.sum)
    
    # Trial 5: Top discriminating cue is Feature 3 (validity 0.75).
    # WADD strongly favors Option A (WADD_A - WADD_B = +2.5).
    mask_t5 = (sum_a == 6) & (sum_b == 2)
    
    # Trial 3: Top discriminating cue is also Feature 3 (validity 0.75).
    # WADD strongly favors Option B (WADD_A - WADD_B = -1.05).
    mask_t3 = (sum_a == 3) & (sum_b == 5)
    
    # Proportion of choosing Option A (response == 0) in these trials
    p_a_t5 = (data.loc[mask_t5, 'response'] == 0).mean()
    p_a_t3 = (data.loc[mask_t3, 'response'] == 0).mean()
    
    return float(p_a_t5 - p_a_t3)
outcome: self_sim=0.1012 (var=0.0728) adversary_sim=0.0100 (var=0.0172) welch_t=+1.521 p=0.1374 (N=25, alpha=0.01) -> reject

[4] rationale: In all trials of the experiment, the top discriminating cue favors Option A. Thus, pure Take The Best (TTB) predicts a constant probability of choosing A across all trials. However, the Advocated Theory posits a fallback to the Weighted Additive (WADD) strategy when the top cue's validity falls below a threshold. In Trials 3 and 5, WADD strongly favors Option A, whereas in Trials 1 and 4, WADD favors Option B. By comparing the rate of choosing A between these two pooled sets of trials (sum_b == 0 vs sum_b >= 2), we compute a metric that pure TTB expects to be zero, while the Advocated Theory expects to be significantly positive. Pooling multiple trials per condition reduces within-subject response noise, leading to tighter per-subject estimates and higher discriminability.
metric_source:
def metric(data: pd.DataFrame) -> float:
    # Calculate the sum of ratings for option B to distinguish trials
    sum_b = data['option_b_ratings'].apply(sum)
    
    # Trials 3 and 5: sum_b == 0. 
    # In these trials, WADD strongly favors Option A.
    mask_wadd_a = (sum_b == 0)
    
    # Trials 1 and 4: sum_b >= 2.
    # In these trials, WADD favors Option B.
    mask_wadd_b = (sum_b >= 2)
    
    # Proportion of choosing Option A (response == 0)
    p_a_wadd_a = (data.loc[mask_wadd_a, 'response'] == 0).mean()
    p_a_wadd_b = (data.loc[mask_wadd_b, 'response'] == 0).mean()
    
    return float(p_a_wadd_a - p_a_wadd_b)
outcome: self_sim=0.1521 (var=0.0815) adversary_sim=0.0058 (var=0.0039) welch_t=+2.503 p=0.01884 (N=25, alpha=0.01) -> reject

## RESPONSE FORMAT
Return a JSON object with the following fields:
- metric: Python source defining def metric(data: pd.DataFrame) -> float.
- rationale: Reasoning behind this metric; used when interpreting results.


## Response

```json
{
  "metric_source": "def metric(data: pd.DataFrame) -> float:\n    # Calculate the sum of ratings for option B to distinguish trials\n    sum_b = data['option_b_ratings'].apply(sum)\n    \n    # Trials 2, 3, and 5: sum_b <= 1.\n    # In these trials, both Take The Best (TTB) and the Weighted Additive (WADD) fallback favor Option A.\n    mask_high = sum_b <= 1\n    \n    # Trial 1: sum_b == 2.\n    # The top discriminating cue is Feature 3 (validity 0.65), which will frequently fall below the validity threshold.\n    # When this happens, the WADD fallback is triggered, which favors Option B, lowering the probability of choosing A.\n    mask_low = sum_b == 2\n    \n    # Proportion of choosing Option A (response == 0)\n    p_a_high = (data.loc[mask_high, 'response'] == 0).mean()\n    p_a_low = (data.loc[mask_low, 'response'] == 0).mean()\n    \n    return float(p_a_high - p_a_low)",
  "rationale": "To maximize the discriminability between the theories and reduce per-subject variance, we must carefully select which trials to pool. Pure TTB predicts an identical, flat probability of choosing Option A across all 5 trials because Option A always wins on the top discriminating cue. The Advocated Theory expects a drop in the probability of choosing A only when the top cue's validity is low enough to trigger the WADD fallback AND WADD favors Option B. Trial 1 perfectly fits this: the top cue is Feature 3 (validity 0.65), and WADD strongly favors B. In contrast, while WADD also favors B in Trial 4, its top cue is Feature 1 (validity 0.90), meaning the WADD fallback is rarely triggered. Previous attempts diluted the effect by pooling Trial 4 with Trial 1. By strictly comparing Trial 1 against a pooled baseline of Trials 2, 3, and 5 (where both TTB and WADD favor A), we maximize the mean difference while significantly reducing the variance of the baseline estimate."
}
```

## Usage

```json
{
  "prompt_token_count": 6651,
  "candidates_token_count": 522,
  "total_token_count": 11226
}
```
